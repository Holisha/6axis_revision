import sys
sys.path.append('..')

from argparse import ArgumentParser
from utils import StorePair

def argument_setting():
    r"""
    return the arguments
    """
    parser = ArgumentParser()

    #################################
    ## demo process setting
    parser.add_argument('--char-max', type=int, default=900,
                        help='set the upper limit of characters number (default: 900)')
    parser.add_argument('--noise-max', type=float, default=10,
                        help='set the maximun amplitude of noise range (default: 10)')

    #################################
    ## dataset setting
    parser.add_argument('--noise', type=float, nargs=2, default=[-1,1],
                        help='set the noise range (default: [-1, 1])')
    parser.add_argument('--stroke-len', type=int, default=150,
                        help='set the length of each stroke (default: 150)')
    
    parser.add_argument('--test-char', type=int, default=None,
                        help='set the character number of the testing data you want to build (default: None)')
    parser.add_argument('--test-num', type=int, default=1,
                        help='set the numbers of the testing datas you want to create (default: 1)')

    # extended length method
    parser.add_argument('--extend', type=str, default='inter',
                        metavar='tail, inter', help="set the complement method (default: 'inter')")

    # dataset file name format
    parser.add_argument('--char-idx', type=int, default=4,
                        help='set the index length of each char of file name (default: 4)')
    parser.add_argument('--stroke-idx', type=int, default=2,
                        help='set the length of each stroke (default: 2)')
    parser.add_argument('--num-idx', type=int, default=4,
                        help='set the length of each stroke (default: 4)')

    # dataset path
    parser.add_argument('--root-path', type=str, default='./dataset/',
                        help='set the root path (default: ./dataset/)')
    parser.add_argument('--input-path', type=str, default='./dataset/6axis/',
                        help='set the path of the original datas (default: ./dataset/6axis/)')
    parser.add_argument('--test-path', type=str, default='./test/',
                        help='set the path of the testing datas (default: ./test/)')
    parser.add_argument('--target-path', type=str, default='./dataset/target/',
                        help="target dataset path (default: './dataset/target/')")

    # for preprocessor setting
    parser.add_argument('--less', action='store_true', default=False,
                        help='get the less of the dataset (default: False)')

    ##########################
    # logger setting
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='set the logger path of pytorch model (default: ./logs)')
    # save setting
    parser.add_argument('--save-path', type=str, default='./output/',
                        help='set the output file (csv or txt) path (default: ./output/)')

	##########################
    # testing args
	# doc setting
    parser.add_argument('--doc', type=str, metavar='../doc/sample.yaml',
                        help='load document file by position(default: None)')

    # dataset setting
    parser.add_argument('--batch-size', type=int, default=1,
                        help='set the batch size (default: 1)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='set the number of processes to run (default: 8)')

    # model setting
    parser.add_argument('--model-name', type=str, default='FSRCNN',
                        metavar='FSRCNN, DDBPN' ,help="set model name (default: 'FSRCNN')")
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')
    parser.add_argument('--model-args', action=StorePair, nargs='+', default={},
                        metavar='key=value', help="set other args (default: {})")
    parser.add_argument('--load', action='store_false', default=True,
                        help='load model parameter from exist .pt file (default: True)')
    parser.add_argument('--version', type=int, dest='load',
                        help='load specific version (default: False)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')

    # loss setting
    parser.add_argument('--alpha', type=float, default=1,
                        help="set loss 1's weight (default: 1)")
    parser.add_argument('--beta', type=float, default=1,
                        help="set loss 2's weight (default: 1)")
    parser.add_argument('--criterion', type=str, default='huber',
                        help="set criterion (default: 'huber')")

    return parser.parse_args()