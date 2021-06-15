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


def full_eval(args=None, train_dir=None):
    if args is None:
        args = command_parser.parse_arguments()

    args.phase = 'eval'
    args.episode_type = 'TestValEpisode'
    args.test_or_val = 'val'

    args.data_dir = os.path.expanduser('~/Data/AI2Thor_offline_data_2.0.2/')

    if args.detection_feature_file_name is None:
        args.detection_feature_file_name = '{}_features_{}cls.hdf5'.format(args.detection_alg, args.num_category)

    start_time_str = time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())
    )
    tb_log_dir = os.path.join(args.work_dir, 'runs', '{}_{}_{}'.format(args.title, args.phase, start_time_str))
    log_writer = SummaryWriter(log_dir=tb_log_dir)

    if train_dir is not None:
        args.results_json = os.path.join(train_dir, 'result.json')

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    # Get all valid saved_models for the given title and sort by train_ep.
    checkpoints = [(f, f.split("_")) for f in os.listdir(args.save_model_dir)]
    checkpoints = [
        (f, int(s[-3]))
        for (f, s) in checkpoints
        if len(s) >= 4 and f.startswith(args.title) and int(s[-3]) >= args.test_start_from
    ]
    checkpoints.sort(key=lambda x: x[1])

    best_model_on_val = None
    best_performance_on_val = 0.0
    for (f, train_ep) in tqdm(checkpoints, desc="Checkpoints."):

        model = os.path.join(args.save_model_dir, f)
        args.load_model = model

        # run eval on model
        args.test_or_val = "val"
        main_eval(args, create_shared_model, init_agent)

        # check if best on val.
        with open(args.results_json, "r") as f:
            results = json.load(f)

        if results["success"] > best_performance_on_val:
            best_model_on_val = model
            best_performance_on_val = results["success"]

        log_writer.add_scalar("val/success", results["success"], train_ep)
        log_writer.add_scalar("val/spl", results["spl"], train_ep)

        if args.include_test:
            args.test_or_val = "test"
            main_eval(args, create_shared_model, init_agent)
            with open(args.results_json, "r") as f:
                results = json.load(f)

            log_writer.add_scalar("test/success", results["success"], train_ep)
            log_writer.add_scalar("test/spl", results["spl"], train_ep)

    args.test_or_val = "test"
    args.load_model = best_model_on_val
    main_eval(args, create_shared_model, init_agent)

    with open(args.results_json, "r") as f:
        results = json.load(f)

    print(
        tabulate(
            [
                ["SPL >= 1:", results["GreaterThan/1/spl"]],
                ["Success >= 1:", results["GreaterThan/1/success"]],
                ["SPL >= 5:", results["GreaterThan/5/spl"]],
                ["Success >= 5:", results["GreaterThan/5/success"]],
            ],
            headers=["Metric", "Result"],
            tablefmt="orgtbl",
        )
    )

    print("Best model:", args.load_model)


if __name__ == "__main__":
    full_eval()