import numpy as np
import csv
from argparse import ArgumentParser


def argument_setting():
    r"""
    return the arguments
    """
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, default='char00312_stroke.txt',
                        help='set the filename that you want to add noises')
    parser.add_argument('--output-num', type=int, default=1000,
                        help='set the numbers of the datas you want to create(default: 1000)')                    
    parser.add_argument('--noise', type=float, nargs=2,default=[-1,1],
                        help='set the noise')

    return parser.parse_args()

