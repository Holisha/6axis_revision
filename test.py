import pandas as pd
import numpy as np
import torch
from  torchvision import transforms

test = pd.read_csv('char00436_stroke.csv', header=None)
arr = test.iloc[:, :-1].to_numpy()

# print(torch.from_numpy(arr))
print(transforms.ToTensor()(arr))