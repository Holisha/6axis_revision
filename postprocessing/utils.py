from argparse import ArgumentParser

def argument_setting():
    r"""
    return arguments
    """
    parser = ArgumentParser()

    parser.add_argument('--path', type=str, default='./output',
                        help='set the data path (default: ./output)')

    return parser.parse_args()