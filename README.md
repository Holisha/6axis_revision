# 6 Axis Revision 

- [6 Axis Revision](#6-axis-revision)
  - [Introduction](#introduction)
  - [Folder structure](#folder-structure)
  - [Requirements and Dependencies](#requirements-and-dependencies)
  - [Demo](#demo)
  - [Training](#training)
    - [model parameter](#model-parameter)
  - [Pretained model](#pretained-model)
  - [Eval](#eval)

## Introduction

- This project is aiming at revise calligraphy words based on CNN method.

## Folder structure

```
   \---6axis_revision
    |   .gitignore
    |   LOG.md
    |   README.md
    |   requirements.txt
    |
    +---dataset
    |   +---6axis
    |   |
    |   +---target
    |   |
    |   +---test
    |   |
    |   +---train
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
    +---output-train
    |
    +---src
        |   .gitignore
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
        +---logs
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
- Demo process includes `Pre-processing` (Building Testing Dataset)、`Evaluating` and `Post-processing`.
- To run the demo of evaluating process
    1. Extract folder `dataset/` in `demo.tar.gz` under `6axis_revision/`
    2. Extract folder `logs/` in `demo.tar.gz` under `demo/`
    3. See [Folder structure](#Folder-structure) to make sure.
    4. Change directory to `demo/`.
    5. Run the following command in shell.
    ```shell
    python demo.py
    ```
    6. Input the character number and the range of noise.
    7. Program will display the test loss on the screen.
- You could add `--gui` argument to run for GUI, also add `--usb-path` to store the demo files to USB or others path . Example:
```shell
python demo.py --gui --usb-path USB_PATH
```
- The evaluating result, including input, output and target of Robot command file, and 2D visualization compare picture, will store in `demo/output/test_char/`.

## Training
- Training process is to train the model.
- To run the training process
    1. Extract folder `train/` in `full-inter-train.tar.gz` under `dataset/`
    2. See [Folder structure](#Folder-structure) to make sure.
    3. Change directory to `src/`.
    4. Run the following command in shell to execute the default environment.
    ```shell
    python train.py --doc ../doc/sample_trainV7.yaml
    ```
    5. Program will display the training loss on the screen.
- The training result, including input, output and target of Robot command file, and 2D visualization compare picture, will store in `output-train/`.
- You could run the flollowing command to track and visualizing metrics such as loss
  ```
  tensorboard --logdir ./logs/
  ```
  - Open http://localhost:6006/ in Web browser.
- To train with argument vector
    ```
    python train.py --gpu-id 0 ...
    ```
- Please check `doc/sample_trainV7.yaml` in detail
    
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
