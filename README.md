# 6 Axis Revision 

<<<<<<< HEAD
- [6 Axis Revision](#6-axis-revision)
  - [Introduction](#introduction)
  - [Folder structure](#folder-structure)
  - [Requirements and Dependencies](#requirements-and-dependencies)
  - [Demo](#demo)
  - [Training](#training)
    - [model parameter](#model-parameter)
  - [Pretained model](#pretained-model)
  - [Eval](#eval)
=======
- [TOC]
- Version: 5.0
    - update at 2020/8/30

>>>>>>> a865e98bdc97a7f22a2ba8d106269b90d5458f3e
## Introduction

- This project is aiming at revise calligraphy words based on CNN method.

## Folder structure

```
   \---6axis_revision
    |   .gitignore
    |   LOG.md
    |   README.md
    |
    +---dataset
    |   +---6axis
    |   |       char00001_stroke.txt
    |   |       ...
    |   |
    |   +---target
    |   |   +---0001
    |   |   |   +---01
    |   |   |           0001_01.npy
    |   |   |
    |   |   +---0002 ...
    |   |
    |   +---test
    |   |   +---0042
    |   |       +---01
    |   |       |       0042_01_0001.npy
    |   |       |
    |   |       +---02 ...
    |   |
    +---demo
    |   |   .gitignore
    |   |   README.md
    |   |   __init__.py
    |   |   bg.qrc
    |   |   demo.py
    |   |   demo.ui
    |   |   demo_utils.py
    |   |   project.py
    |   |   
    |   +---calligraphy
    |   |   |   __init__.py
    |   |   |   calligraphy_transform.py
    |   |   |   char_list.csv
    |   |   |   code.py
    |   |   |   
    |   |   +---utils
    |   |           __init__.py
    |   |           tools.py
    |   |           
    |   +---imgs
    |   |       ...
    |   |       
    |   +---logs
    |       +---FSRCNN
    |           +---version_0
    |                   FSRCNN_1x.pt
    |
    +---doc
    |       sampleV4.json
    |       sampleV4.yaml
    |       
    +---src
        |   __init__.py
        |   dataset.py
        |   eval.py
        |   test_error.py
        |   train.py
        |   train_error.py
        |   train_recurrent.py
        |   utils.py
        |   
        +---light
        |       fsrcnn.py
        |       
        +---model
        |   |   __init__.py
        |   |   loss.py
        |   |   optimizer.py
        |   |   
        |   +---DBPN
        |   |       __init__.py
        |   |       models.py
        |   |           
        |   +---FSRCNN
        |           __init__.py
        |           models.py
        |           
        +---postprocessing
        |       .gitignore
        |       README.md
        |       __init__.py
        |       axis2img.py
        |       csv2txt.py
        |       post_utils.py
        |       postprocessor.py
        |       stroke2char.py
        |       verification.py
        |       
        +---preprocessing
                .gitignore
                __init__.py
                pre_utils.py
                preprocessor.py
                readme.md
```


## Requirements and Dependencies

- This project runs on gpu only
- CUDA 10.1
- Python 3.6 or upper version
    ```shell
    pip install requirements.txt
    ```

## Demo

- To run the demo of evaluating process
    1. Extract folder `dataset/` in `demo.tar.gz` under `6axis_revision/`
    2. Extract folder `logs/` in `demo.tar.gz` under `demo/`
    3. Change directory to `demo/`.
    4. Run the following command in shell.
    ```shell
    python demo.py
    ```
    5. Input the character number and the range of noise.
    6. Program will display the test loss on the screen.
- You could add `--gui` argument to run for GUI, also add `--usb-path` to store the demo files to USB or others path . Example:
```shell
python demo.py --gui --usb-path USB_PATH
```
- The evaluating result, including input, output and target of Robot command file, and 2D visualization compare picture, will store in `demo/output/test_char/`.

## Training

- To train with document file
    ```
    python train.py --doc
    ```

- To train with argument vector
    ```
    python train.py --gpu-id 0 ...
    ```

- please check `doc/sampleV4.yaml` in detail

### model parameter

- learning rate:
    - FSRCNN: 1e-3
    - DBPN: 1e-4

```python
DBPN-S:
	stages=2,
	n0=128,
	nr=32

DBPN-SS:
	stages=2,
	n0=64,
	nr=18
	
ADBPN:
	col_slice=3,
	stroke_len=150

ADBPN-S:
	stages=2,
	n0=128,
	nr=32
	
ADBPN-SS:
	stages=2,
	n0=64,
	nr=18
```

## Pretained model

- Developing

| Network | Task      | Download |
| ------- | --------- | -------- |
| FSRCNN  | 史        | None     |
| FSRCNN  | 900 words | None     |
| D-DBPN  | 900 words | None     |

## Eval

- To evaluate the model by
`python test.py --gpu-id 0 ...`