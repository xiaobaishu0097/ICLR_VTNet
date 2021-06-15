import os
import datetime
import random
import ctypes
import setproctitle
import time
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

from models import PreTrainedVisualTransformer
from runners.pretrain_engine import train_one_epoch, evaluate
from utils import command_parser
from utils.misc_util import Logger
from utils.class_finder import model_class
from utils.pretrain_util import PreVisTranfsDataset, init_distributed_mode
import utils.misc as utils

os.environ["OMP_NUM_THREADS"] = "1"

def main():
    setproctitle.setproctitle("Training")
    args = command_parser.parse_arguments()
    init_distributed_mode(args)

    if args.title == 'a3c':
        args.title = 'pretrain_vistrans'
    if args.model == 'BaseModel':
        args.model = 'PreTrainedVisualTransformer'
    args.data_dir = os.path.expanduser('~/Data/AI2Thor_VisTrans_Pretrain_Data/')

    print(args)

    # records related
    start_time_str = time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())
    )
    work_dir = os.path.join(args.work_dir, '{}_{}_{}'.format(args.title, args.phase, start_time_str))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not args.no_logger:
        log_file = os.path.join(work_dir, 'pretrain.txt')
        sys.stdout = Logger(log_file, sys.stdout)
        sys.stderr = Logger(log_file, sys.stderr)

    # tb_log_dir = os.path.join(args.work_dir, 'runs', '{}_{}_{}'.format(args.title, args.phase, start_time_str))
    # log_writer = SummaryWriter(log_dir=tb_log_dir)

    save_model_dir = os.path.join(work_dir, 'trained_models')
    args.save_model_dir = save_model_dir
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # start training preparation steps
    if args.remarks is not None:
        print(args.remarks)
    print('Training started from: {}'.format(
        time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    )

    device = torch.device('cuda')

    model_creator = model_class(args.model)
    model = model_creator(args)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = PreVisTranfsDataset(args, 'train')
    dataset_val = PreVisTranfsDataset(args, 'val')
    dataset_test = PreVisTranfsDataset(args, 'test')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_train = torch.utils.data.WeightedRandomSampler([1, 1, 1, 1, 1, 1], len(dataset_train))
    sampler_val = torch.utils.data.RandomSampler(dataset_val)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, num_workers=args.num_workers)

    if args.load_model is not None:
        checkpoint = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        epoch = args.start_epoch
        evaluate(model, criterion, data_loader_test, device, epoch, args.record_act_map)
        return 0

    print('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,
            print_freq=args.print_freq)
        lr_scheduler.step()

        checkpoint_paths = [os.path.join(save_model_dir, 'checkpoint.pth')]
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.epoch_save == 0:
            print('Evaluating on Test dataset!')
            evaluate(model, criterion, data_loader_test, device, epoch)
            checkpoint_paths.append(os.path.join(save_model_dir, f'checkpoint{epoch:04}.pth'))

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        print('Evaluating on Val dataset!')
        evaluate(model, criterion, data_loader_val, device, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
