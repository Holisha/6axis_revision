import pandas as pd
import numpy as np
import os

def stroke_statistics(path='6d/'):
    max_cnt = np.int64(0)
    mean_cnt = np.int64(0)

    for dir_name in os.listdir(path):
        df = pd.read_csv(
            os.path.join(path, dir_name, dir_name + '.csv'),
            header=None
        )

        tmp = df.groupby(6)[0].count()

        if tmp.max() > max_cnt:
            max_cnt = tmp.max()
        
        if tmp.mean() > mean_cnt:
            mean_cnt = tmp.mean()
    
    print(f'max stroke number:{max_cnt}\nmean stroke number:{mean_cnt}')


if __name__ == '__main__':
    stroke_statistics()