# 6 Axis Revision 

## Dependencies
- CUDA 10.1
- Python 3

## Python Packages Installation
PyTorch, see: https://pytorch.org/get-started/locally/

The others:
```
pip install -r requirements.txt
```

## Getting Started
There are two methods to get started. The first one is running ```make``` to finish all the processes. The second one is step-by-step.

### Method 1
Run ```make``` to training Model, testing and change 6-axis to picture.
```bash
make
```

### Method 2: Step 1. Training Model And Testing
Training learning model and testing. It will output totally 10 ```.csv``` files from the training inputs, results and ground-truth per (Epoch / 10) times.

```shell
python main.py
```

Output csv files will save in the directory ```output/```.

#### Change Epoch Number
If you want to change epoch number of model, you can edit line **14** in ```main.py``` and line **7** in ```axis2img.py```. (Default epoch number is 10)

```python
NUM_EPOCHS = 10     # number of epochs
```

### Method 2: Step 2. Change 6-Axis to Picture
Compare three 6-axis ```.csv``` file of each epoch, and save results to ```.png``` file.

```bash
python axis2img.py
```

Output pictures will save in the directory ```output/```, and the pictures' name will be started by **epoch_**.