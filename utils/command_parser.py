import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visual Navigation')

    #TODO: remove this argument in the future
    parser.add_argument(
        '--phase',
        type=str,
        default='train',
        help='train, eval or debug three choices'
    )

    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=30,
        metavar='M',
        help='maximum length of an episode (default: 100)',
    )

    parser.add_argument(
        '--shared-optimizer',
        default=True,
        metavar='SO',
        help='use an optimizer with shared statistics.',
    )


    parser.add_argument(
        '--optimizer',
        default='SharedAdam',
        metavar='OPT',
        help='shared optimizer choice of SharedAdam or SharedRMSprop',
    )

    parser.add_argument(
        '--amsgrad',
        default=True,
        metavar='AM',
        help='Adam optimizer amsgrad parameter'
    )

    parser.add_argument(
        '--grid_size',
        type=float,
        default=0.25,
        metavar='GS',
        help='The grid size used to discretize AI2-THOR maps.',
    )

    parser.add_argument(
        '--docker_enabled',
        action='store_true',
        help='Whether or not to use docker.'
    )

    parser.add_argument(
        '--x_display',
        type=str,
        default=None,
        help='The X display to target, if any.'
    )

    parser.add_argument(
        '--test_timeout',
        type=int,
        default=10,
        help='The length of time to wait in between test runs.',
    )

    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='If true, output will contain more information.',
    )

    parser.add_argument(
        '--train_thin',
        type=int,
        default=1000,
        help='How often to print'
    )

    parser.add_argument(
        '--num-memory-block',
        type=int,
        default=2,
        help='the number of memory blocks'
    )

    parser.add_argument(
        '--local_executable_path',
        type=str,
        default=None,
        help='a path to the local thor build.',
    )

    parser.add_argument(
        '--hindsight_replay',
        type=bool,
        default=False,
        help='whether or not to use hindsight replay.',
    )

    parser.add_argument(
        '--enable_test_agent',
        action='store_true',
        help='Whether or not to have a test agent.',
    )

    parser.add_argument(
        '--train_scenes',
        type=str,
        default='[1-20]',
        help='scenes for training.'
    )

    parser.add_argument(
        '--val_scenes',
        type=str,
        default='[21-30]',
        help='old validation scenes before formal split.',
    )

    parser.add_argument(
        '--possible_targets',
        type=str,
        default='FULL_OBJECT_CLASS_LIST',
        help='all possible objects.',
    )

    # if none use all dest objects
    parser.add_argument(
        '--train_targets',
        type=str,
        default=None,
        help='specific objects for this experiment from the object list.',
    )

    parser.add_argument(
        '--action_space',
        type=int,
        default=6,
        help='space of possible actions.'
    )

    parser.add_argument(
        '--attention-sz',
        type=int,
        default=512,
    )

    parser.add_argument(
        '--compute_spl',
        action='store_true',
        help='compute the spl.'
    )

    parser.add_argument(
        '--include-test',
        action='store_true',
        help='run test during eval'
    )

    parser.add_argument(
        '--disable-strict_done',
        dest='strict_done',
        action='store_false'
    )

    parser.set_defaults(strict_done=True)

    parser.add_argument(
        '--results-json',
        type=str,
        default='metrics.json',
        help='Write the results.'
    )

    parser.add_argument(
        '--visualize-file-name',
        type=str,
        default='visual_temp.json'
    )

    parser.add_argument(
        '--agent_type',
        type=str,
        default='NavigationAgent',
        help='Which type of agent. Choices are NavigationAgent or RandomAgent.',
    )

    parser.add_argument(
        '--episode_type',
        type=str,
        default='BasicEpisode',
        help='Which type of agent. Choices are NavigationAgent or RandomAgent.',
    )

    parser.add_argument(
        '--fov',
        type=float,
        default=100.0,
        help='The field of view to use.'
    )

    parser.add_argument(
        '--scene-types',
        nargs='+',
        default=['kitchen', 'living_room', 'bedroom', 'bathroom'],
    )

    parser.add_argument(
        '--gradient_limit',
        type=int,
        default=4,
        help='How many gradient steps allowed for MAML.',
    )

    parser.add_argument(
        '--test_or_val',
        default='val',
        help='test or val'
    )

    parser.add_argument(
        '--test-start-from',
        type=int,
        default=0,
        help='start from given episode'
    )

    parser.add_argument(
        '--multi-heads',
        type=int,
        default=1
    )

    parser.add_argument(
        '--keep-ori-obs',
        action='store_true',
    )

    parser.add_argument(
        '--record-attention',
        action='store_true',
    )

    parser.add_argument(
        '--test-speed',
        action='store_true'
    )

    # ==================================================
    # arguments with normal settings
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='BaseModel',
        help='Model to use.'
    )

    parser.add_argument(
        '--eval',
        action='store_true',
        help='run the test code'
    )

    parser.add_argument(
        '--record-act-map',
        action='store_true',
    )

    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=-1,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)',
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=12,
        metavar='W',
        help='how many training processes to use (default: 4)',
    )

    parser.add_argument(
        '--title',
        type=str,
        default='a3c',
        help='Info for logging.'
    )

    parser.add_argument(
        '--work-dir',
        type=str,
        default='./debugs/',
        help='Work directory, including: tensorboard log dir, log txt, trained models',
    )

    parser.add_argument(
        '--save-model-dir',
        default='debugs',
        help='folder to save trained navigation',
    )

    parser.add_argument(
        '--max-ep',
        type=float,
        default=6000000,
        help='maximum # of episodes'
    )

    parser.add_argument(
        '--ep-save-freq',
        type=int,
        default=1e5,
        help='save model after this # of training episodes (default: 1e+4)',
    )

    parser.add_argument(
        '--test-after-train',
        action='store_true',
        help='run test after training'
    )

    parser.add_argument(
        '--remarks',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--no-logger',
        action='store_true',
    )

    # arguments related with continue training based on existed trained models
    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help='Path to load a saved model.'
    )

    parser.add_argument(
        '--continue-training',
        type=str,
        default=None,
        help='continue training based on given model'
    )

    parser.add_argument(
        '--fine-tuning',
        type=str,
        default=None,
        help='fine tune based on given model'
    )

    parser.add_argument(
        '--pretrained-trans',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--warm-up',
        action='store_true',
    )

    # arguments related with pretraining Visual Transformer
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
    )

    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
    )

    parser.add_argument(
        '--epoch-save',
        type=int,
        default=5,
    )

    # ==================================================
    # arguments related with data
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/data/AI2Thor_Dataset/',
        help='where dataset is stored.',
    )

    parser.add_argument(
        '--num-category',
        type=int,
        default=22,
        help='the number of categories'
    )

    parser.add_argument(
        '--graph-file',
        type=str,
        default='graph.json',
        help='the name of the graph file'
    )

    parser.add_argument(
        '--grid-file',
        type=str,
        default='grid.json',
        help='the name of the grid file'
    )

    parser.add_argument(
        '--visible-map-file-name',
        type=str,
        default='visible_object_map_1.5.json',
        help='the name of the visible object map file'
    )

    parser.add_argument(
        '--detection-alg',
        type=str,
        default='detr',
        choices=['fasterrcnn', 'detr', 'fasterrcnn_bottom']
    )

    parser.add_argument(
        '--detection-feature-file-name',
        type=str,
        default=None,
        help='Which file store the detection feature?'
    )

    parser.add_argument(
        '--images-file-name',
        type=str,
        default='resnet18_featuremap.hdf5',
        help='Where the controller looks for images. Can be switched out to real images or Resnet features.',
    )

    parser.add_argument(
        '--optimal-action-file-name',
        type=str,
        default='optimal_action.json'
    )

    # ==================================================
    # arguments related with models
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        metavar='LR',
        help='learning rate (default: 0.0001)',
    )

    parser.add_argument(
        '--pretrained-lr',
        type=float,
        default=0.00001,
    )

    parser.add_argument(
        '--wo-location-enhancement',
        action='store_true',
    )

    parser.add_argument(
        '--weight-decay',
        default = 1e-4,
        type = float,
    )

    parser.add_argument(
        '--inner-lr',
        type=float,
        default=0.0001,
        metavar='ILR',
        help='learning rate (default: 0.01)',
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        metavar='G',
        help='discount factor for rewards (default: 0.99)',
    )

    parser.add_argument(
        '--tau',
        type=float,
        default=1.00,
        metavar='T',
        help='parameter for GAE (default: 1.00)',
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=1e-2,
        help='entropy regularization term'
    )

    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0,
        help='The dropout ratio to use (default is no dropout).',
    )

    parser.add_argument(
        '--hidden-state-sz',
        type=int,
        default=512,
        help='size of hidden state of LSTM.'
    )

    parser.add_argument(
        '--nhead',
        type=int,
        default=8,
    )

    parser.add_argument(
        '--lstm-nhead',
        type=int,
        default=4,
    )

    parser.add_argument(
        '--num-encoder-layers',
        type=int,
        default=6,
    )

    parser.add_argument(
        '--num-decoder-layers',
        type=int,
        default=6,
    )

    parser.add_argument(
        '--dim-feedforward',
        type=int,
        default=512,
    )

    parser.add_argument(
        '--il-rate',
        type=float,
        default=0.1,
        help='the rate of imitation learning loss'
    )

    parser.add_argument(
        '--action-embedding-before',
        action='store_true',
    )

    parser.add_argument(
        '--multihead-attn-gates',
        type=str,
        default=['input', 'cell'],
        nargs='+',
    )

    parser.add_argument(
        '--replace-input-gate',
        action='store_true',
    )

    parser.add_argument(
        '--lr-drop-weight',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--lr-drop-eps',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--lr-min',
        type=float,
        default=0.0001,
    )

    # arguments related with pretraining Visual Transformer
    parser.add_argument(
        '--lr-drop',
        default=10,
        type=int
    )

    parser.add_argument(
        '--clip_max_norm',
        default=0.1,
        type=float,
        help='gradient clipping max norm'
    )

    parser.add_argument(
        '--print-freq',
        type=int,
        default=500,
    )

    # ==================================================
    # arguments related with DETR detector
    parser.add_argument(
        '--detr',
        action='store_true',
    )

    parser.add_argument(
        '--detr-padding',
        action='store_false',
        help='padding non-object classes in detr detection featuers with 0'
    )

    args = parser.parse_args()

    return args
