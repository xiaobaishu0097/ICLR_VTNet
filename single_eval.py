from __future__ import print_function, division
import os
import json
import time

from utils import command_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval
from tqdm import tqdm
from tabulate import tabulate

from tensorboardX import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"


def single_eval(args=None):
    if args is None:
        args = command_parser.parse_arguments()

    args.phase = 'eval'
    args.episode_type = 'TestValEpisode'
    args.test_or_val = 'test'

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    checkpoint = args.load_model

    if args.detection_feature_file_name is None:
        args.detection_feature_file_name = '{}_features_{}cls.hdf5'.format(args.detection_alg, args.num_category)

    model = os.path.join(args.save_model_dir, checkpoint)
    args.load_model = model

    # run eval on model
    # args.test_or_val = "val"
    main_eval(args, create_shared_model, init_agent)


if __name__ == "__main__":
    single_eval()