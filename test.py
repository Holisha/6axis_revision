import pandas as pd
import numpy as np


test = pd.read_csv('char00436_stroke.csv', header=None)
tmp = test.groupby(6)[0].count().max()

print(tmp)
print(type(tmp))



""" import pandas as pd
import os
from glob import glob


def stroke_statistics(path='6d/', rulke):
    


if __name__ == '__main__':
    stroke_statistics() """
