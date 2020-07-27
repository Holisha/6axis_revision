# 6 Axis Revision 

## Folder structure

```
6axis_revision
|   .gitignore
|   axis2img.py
|   dataset.py
|   get_6d_3d.py
|   README.md
|   stroke2char.py
|   test.py
|   test_error.py
|   tmp.py
|   tmp.txt
|   train.py
|   train_error.py
|   train_recurrent.py
|   utils.py
|   __init__.py
|   
+---axis2result
+---light
|       fsrcnn.py
|       
+---model
|   |   loss.py
|   |   __init__.py
|   |   
|   \---FSRCNN
|           models.py
|           __init__.py
|           
\---preprocess
    |   .gitignore
    |   preprocess.py
    |   readme.md
    |   requirement.txt
    |   utils.py
    |   
    \---6axis
```

## Commit format

- `Fixed`: **where**, solve **what bug**, **how**
- `New`: **Where**, Add new feature
- `Other`: Special case

## TODO

- update readme file

## Dependencies
- CUDA 10.1
- Python 3.6 or upper version

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