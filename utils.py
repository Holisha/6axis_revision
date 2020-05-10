from argparse import ArgumentParser


def argument_setting():
    parser = ArgumentParser()
    parser.add_argument('--train-path', type=str, default='./train')
    parser.add_argument('--test-path', type=str, default='./test')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)

    return parser.parse_args()


