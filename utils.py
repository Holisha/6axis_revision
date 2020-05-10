import numpy as np
import csv
from argparse import ArgumentParser


def argument_setting():
    r"""
    return the arguments
    """
    parser = ArgumentParser()

    # data pre-processing
    parser.add_argument('--stroke-length', type=int, default=59,
                        help='control the stroke length (default: 59)')
    parser.add_argument('--check_interval', type=int, default=100,
                        help='setting output a csv file every epoch of interval (default: 100)')

    # model setting
    parser.add_argument('--light', action='store_true', default=False,
                        help='train by pytorch-lightning model (default: False)')
    parser.add_argument('--train-path', type=str, default='./train',
                        help='training dataset path (default: ./train)')
    parser.add_argument('--test-path', type=str, default='./test',
                        help='test dataset path (default: ./test)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='set the batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='set the number of processes to run (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='set the learning rate (default: 1e-3)')
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='set the epochs (default: 50)')

    # logger setting
    parser.add_argument('--log-path', type=str, default='./logs/FSRCNN',
                        help='set the logger path of pytorch model (default: ./logs/FSRCNN)')
    parser.add_argument('--light-path', type=str, default='./logs/FSRCNN_light',
                        help='set the logger path of pytorch-lightning model (default: ./logs/FSRCNN_light)')

    return parser.parse_args()


def out2csv(inputs, file_string, stroke_length):
    """
    store input to csv file.

    input: tensor data, with cuda device and size = [batch 1 STROKE_LENGTH 6]
    file_string: string, filename

    no output
    """
    output = np.squeeze(inputs.cpu().detach().numpy())
    table = output[0]

    # with open('output/' +  file_string + '.csv', 'w', newline='') as csvfile:
    with open(f'output/{file_string}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(stroke_length):
            row = [] * 7
            row[1:6] = table[i][:]
            row.append('stroke' + str(1))
            writer.writerow(row)
