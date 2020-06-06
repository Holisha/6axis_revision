import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from glob import glob

# PATH = r'char00436_stroke'
PATH = r'output'


def angle2deg(angle):
    return angle * np.pi / 180


def get_3d(path, length=185):
    """
    input: 6 axis csv file path
    path: file path
    length: brush length (default: 185)

    output: (x, y, z) visualized data (type: list)
    """
    data = []
    csv_file = pd.read_csv(path, header=None)
    for row in csv_file.index:
        x = csv_file.iloc[row, 0]
        y = csv_file.iloc[row, 1]
        z = csv_file.iloc[row, 2]
        a = csv_file.iloc[row, 3]
        b = csv_file.iloc[row, 4]
        c = csv_file.iloc[row, 5]
        n_stroke = int(re.search(r'\d+$', csv_file.iloc[row, 6]).group())

        a = angle2deg(a)
        b = angle2deg(b)
        c = angle2deg(c)
        # print(f'{n_stroke}: {x}, {y}, {z}, {a}, {b}, {c}')

        R_a = np.array([
            [1, 0, 0],
            [0, np.cos(a), -1 * np.sin(a)],
            [0, np.sin(a), np.cos(a)]
        ])
                     
        R_b = np.array([
            [np.cos(b), 0, np.sin(b)],
            [0, 1, 0],
            [-1 * np.sin(b), 0, np.cos(b)]
        ])

        R_c = np.array([
            [np.cos(c), -1 * np.sin(c), 0],
            [np.sin(c), np.cos(c), 0],
            [0, 0, 1]
        ])

        # R = Rc * Rb * Ra
        R = np.dot(
            np.dot(R_c, R_b),
            R_a
        )

        A = np.array([
            [R[0, 0], R[0, 1], R[0, 2], x],
            [R[1, 0], R[1, 1], R[1, 2], y],
            [R[2, 0], R[2, 1], R[2, 2], z],
            [0, 0, 0, 1]
        ])

        B = np.identity(4)
        B[2, 3] = length

        T = np.dot(A, B)

        data.append([T[0, 3], T[1, 3], T[2, 3], n_stroke])

    return data


def vis_2d(data):
    """ 
    input: xyz data, character name
    data: x, y, z, stroke num (n * 4)
    output: img of each stroke
    """
    data = np.array(data).reshape(-1, 4)
    tmp_x, tmp_y = [], []
    x, y = [], []
    previous_stroke = 1 # control figure
    for row in range(data.shape[0]):
        # z > 5 mean the brush is dangling
        if data[row, 2] > 5:
            continue
        # if the current stroke is same as preivous stroke
        elif data[row, 3] == previous_stroke:
            tmp_x.append(data[row, 0])
            tmp_y.append(data[row, 1])
            
            x.append(data[row, 0])
            y.append(data[row, 1])
        elif tmp_x and tmp_y:
            plt.figure(previous_stroke)
            plt.plot(tmp_x, tmp_y, color='black')
            plt.title(f'{int(previous_stroke)}')
            plt.savefig(f'{int(previous_stroke)}.png')
            plt.close()

            previous_stroke = data[row, 3]
            tmp_x, tmp_y = [], []

    plt.figure(previous_stroke + 1)
    plt.plot(tmp_x, tmp_y, color='black')
    plt.title(f'{int(previous_stroke)}')
    plt.savefig(f'{int(previous_stroke)}.png')
    plt.close()
    
    plt.figure(previous_stroke + 2)
    plt.plot(x, y, color='black')
    plt.title('all')
    plt.savefig('all.png')
    plt.close()


def vis_2d_compare(target, inputs, outputs, idx=1):
    r"""
    compare only one stroke
    input: xyz data, character name
    data: x, y, z, stroke num (n * 4)
    output: img of each stroke
    """

    inputs = np.array(inputs).reshape(-1, 4)
    outputs = np.array(outputs).reshape(-1, 4)
    target = np.array(target).reshape(-1, 4)

    inputs_x, inputs_y = [], []
    outputs_x, outputs_y = [], []
    target_x, target_y = [], []
    for row in range(target.shape[0]):
        # z > 5 mean the brush is dangling
        if target[row, 2] > 5:
            continue

        inputs_x.append(inputs[row, 0])
        inputs_y.append(inputs[row, 1])

        outputs_x.append(outputs[row, 0])
        outputs_y.append(outputs[row, 1])

        target_x.append(target[row, 0])
        target_y.append(target[row, 1])

    # plot line
    plt.plot(inputs_x, inputs_y, color='blue', label='noised data')
    plt.plot(outputs_x, outputs_y, color='black', label='revised data')
    plt.plot(target_x, target_y, color='red', label='ground truth')

    # save image
    plt.legend(loc='best')
    plt.title(f'output/{idx}_compare')
    plt.savefig(f'output/{idx}_compare.png')

    # plt.show()
    plt.close()


def main():
    # axis2img training outputs
    csv_list = set([int(re.search(r'\d+', file_name).group()) for file_name in glob(os.path.join(PATH, '*00_*.csv'))])

    for file_idx in sorted(csv_list):

        # get 3d data from csv
        target  = get_3d(os.path.join(PATH, f'{file_idx}_target.csv'))
        inputs  = get_3d(os.path.join(PATH, f'{file_idx}_input.csv'))
        outputs = get_3d(os.path.join(PATH, f'{file_idx}_output.csv'))

        # get visual 2d img from ndarray
        vis_2d_compare(
            target=target,
            inputs=inputs,
            outputs=outputs,
            idx=f'epoch_{file_idx}'
        )

    # axis2img test outputs
    csv_list = set([int(re.search(r'\d+', file_name).group()) for file_name in glob(os.path.join(PATH, 'test_*.csv'))])
    for file_idx in sorted(csv_list):
        print(f'test file indx = {file_idx}')
        # get 3d data from csv
        target  = get_3d(os.path.join(PATH, f'test_{file_idx}_target.csv'))
        inputs  = get_3d(os.path.join(PATH, f'test_{file_idx}_input.csv'))
        outputs = get_3d(os.path.join(PATH, f'test_{file_idx}_output.csv'))

        # get visual 2d img from ndarray
        vis_2d_compare(
            target=target,
            inputs=inputs,
            outputs=outputs,
            idx=f'test_{file_idx}'
        )

    # path = PATH + '.csv'
    # data = get_3d(path)
    # vis_2d(data)


if __name__ == '__main__':
    main()
