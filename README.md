# 6 Axis Revision 

## Introduction

- This project is aiming at revise calligraphy words based on CNN method.

## Folder structure

```
   \---6axis_revision
    |   .gitignore
    |   __init__.py
    |   LOG.md
    |   README.md
    |   axis2img.py
    |   dataset.py
    |   get_6d_3d.py
    |   s.txt
    |   test.py
    |   test_error.py
    |   tmp.py
    |   train.py
    |   train_error.py
    |   train_recurrent.py
    |   utils.py
    |   
    +---doc
    |       sampleV4.json
    |       sampleV4.yaml
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
    |              
    |           
    +---postprocessing
    |       .gitignore
    |       __init__.py
    |       README.md
    |       axis2img.py
    |       csv2txt.py
    |       postprocessor.py
    |       post_utils.py
    |       stroke2char.py
    |       
    +---preprocessing
            .gitignore
            preprocess.py
            readme.md
            requirement.txt
            utils.py
```

## TODO

- update readme file
- add pretain model
- add visualization image
- add demo
- add guide file

## Requirements and Dependencies

- This project runs on gpu only
- `pip install requirements.txt`
- CUDA 10.1
- Python 3.6 or upper version

## Training

- To train with document file
`python train.py --doc`

- To train with argument vector
`python train.py --gpu-id 0 ...`

- please check `doc/sample` in detail

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
}
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

- To visualize model's prediction
    - To be implement