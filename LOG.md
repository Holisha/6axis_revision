# project log

- [toc]

## event type:

- format: `event`: `where`: `what`
	- ignore `event` case

- `update`: update **existed** object
	- ex: rename file or variable
- `new`: **new** features
- `fixed`: **bug fixed**
- `bug`: bug found but **yet fix**
- `other`: TBD

## Version 6

### 9/12 - jefflin

- Add extend stroke lenght by interpolation in preprocessing
    - `new`: Add extend stroke lenght by interpolation in preprocessing, 
        - and add noise to the tail
    - `update`: In .gitignore: delete `stroke_statistics` function

### 9/11 - fan

- Add new mechanic in ADBPN:
	- `new`: Add weighted features
	- `update`: update DBPN series coding style
	- `update`: ADBPN now is use channel feature instead of col features
		- to call col features, set `atten=col`

### 9/3 - Angelowen

- adding the content loss to new model training
	- `update`: In train_error.py: add `content_loss` to MSE loss value 
	- `update`: In train_error.py: add `pbar_postfix['Content loss']` 
	- `update`: In train_error.py: add `args.scheduler` and `pbar_postfix['lr']` to record learning rate

### 9/2 - jefflin

- Change the strategy of Early-Stop
	- `update`: In train.py and train_error.py: change `threshold` argument default value and change the order of parameters in `EarlyStopping`
	- `update`: In `EarlyStopping` in utils.py: Change the strategy of Early-Stop. eg, val_loss decrease less than 0.1% (default)
	- `update`: In utils.py: Delete the comment of `csv2txt`

### 8/31 - fan

- Now config will save as original document file when call args.doc
	- `new`: In utils.py: now config will save in version directory with original format
		`.json` -> `.json`, `.yaml` -> `.yaml`
	- `update`: In README.md, add default lr which is based on paper

### 8/30 - Angelowen

- fix fix DBPN/init.py bug and train_error bug
  - `fixed`: train_error.py: fix out2csv bug
  - `fixed`: In DBPN/init.py add ADBPN to call 
  - `update`: In .gitignore: Add `output-v5-1c779cab/` folder

### 8/30 - fan

- Add new model ADBPN and remove output dir
	- `new`: In model/DBPN/models.py: add new model ADBPN
	- `new`: In utils.py: now can call new model by `--model-name ADBPN`
		- model args almost same as DBPN but add `col_slice`, `stroke_len`
	- `removed`: remove output directory
	- `update`: add model args in readme.md

### 8/27 - jeff

- fix lr_scheduler bug and move out2csv place to decrease the I/O times
  - `fixed`: In train.py and train_error.py: fix lr_scheduler bug
  - `update`: In train.py: move out2csv to the end of epoch loop to decrease the I/O times
  - `fixed`: In utils.py in `writer_builder`: check the log_root exists, or create a new one
  - `update`: In .gitignore: Add `output*/` folder

### 8/23 - angelo

- fix line39 bug
  - `fixed`: In train_error.py line39, fix bug

### 8/24 - fan

- Fix iteration bug
	- `fixed`: In train.py, fixed iteration count bug

- Now can view current learing rate with lr_scehduler
	- `fixed`: In utils.py, in `writer_builder`, sort variable `version` now
	- `fixed`: In train.py, add pbar_postfix to control postfix value

### 8/22 - fan

- Now can call dbpn by `--model-name dbpn`
	- `new`: In model/DBPN.py, Add DBPN 
	- `new`: In utils.py, add `dbpn` in model_builder
- `update`: In model/DBPN.py, DBPN series can instantiate normally when scale factor is greater than 1

```python
DBPN-S:
	stages=2,
	n0=64,
	nr=18

DBPN-SS:
	stages=2,
	n0=128,
	nr=32
```

### 8/17 - fan

- Now model parameter and config would be save in log/\*/version\*/
	- `update`: In utils.py: `writer_builder` now would return writer and store path
		- return value: `writer` -> `(writer, model_path)`
	- `update`: In train.py: model_path would be define by `writer_builder`
	- `update`: In utils.py: `model_config` will save config to version directory
	- `update`: In train.py: progress bar can show current learning rate
	- `new`: In loss.py: add rmse loss function
	- `fixed`: In train.py: iteration will show value normaly

### 8/14 - fan

- Now can store model without early stopping
- Add learning rate scheduler and add huber loss function
- modified writer every iteration and epoch
	- `new`: In train.py: add new argument to adjust new builder
	- `fixed`: In train.py: now can store without earlystopping
	- `new`: In utils.py: add criterion builder
		- `--criterion`: choose loss function
	- `new`: In utils.py: add scheduler builder
		- `--scheduler`: to choose step or multiple step scheduler
		- `--lr-method`: to choose scheduler
		- `--step`: set update step(epoch)
		- `--factor`: set decreate factor

- TODO: modified out2csv timing

### 8/11 - jeff

- Add getting more model data from each epoch.
Update: In train.py:
		Comment target scaler and uncomment pred denormalize
Update: In sampleV3.yaml: Update arguments value
Add: In .gitignore: Add output_tmp/ for testing data
Add: In train.py:
		Add argument 'out-num' to set the number of model data to get
Add: In train.py and out2csv in utils.py:
		Add getting more model data from each epoch

- Add getting more model data from each epoch.

Update: In train.py:
		Comment target scaler and uncomment pred denormalize
Update: In sampleV3.yaml: Update arguments value
Add: In .gitignore: Add output_tmp/ for testing data

Add: In train.py:
		Add argument 'out-num' to set the number of model data to get
Add: In train.py and out2csv in utils.py:
		Add getting more model data from each epoch
		
### 8/11 - fan

- Dataset cross validation big update
	- New: In dataset.py: add new function 'cross_validation', split data by randomsplit instead sampler
	- Rename: In dataset.py: 'cross_validation' -> '_cross_validation'
	- Update: In train*: update loading dataset to fit in new CV function
- now different dataset have own progress bar but leave train set only. Update doc
	- Update: In train.py progress modified
	- Update: update doc parameters

### 8/10 - fan

- compression tensor to range `[-1, 1]`

### 8/09 - jeff

- Add modularization of postporcessing, early-stop threshold argument, out2csv taking specific data

1. Modularize postporcessing and add into train process
2. Add Early-Stop threshold argument
3. Add out2csv take specific valid data
4. Fix out2csv bug about output path

New: In train.py and train_error.py: Add postprocessing in the end
New: In train.py and EarlyStopping in utils.py: Add argument threshold
Update: In train.py: Move out2csv from train section to valid section
Update: In utils.py: Fix bug about output path
Update: In dataset.py: Change valid Sampler to SequentialSampler
Comment: In utils.py: Comment csv2txt function and move to the end
Update: In postprocessing/: Modularization

### 8/08 - fan

- Model-args now can be assign value by pairs
	- Update: In doc: update sample files which are available to fit in current version
	- New: In utils.py: Add StorePair to custom argparse.action object
	- Update: In train.py and test.py: modified model-args from list to dict, namely pass kwargs to model_builder
	- Comment: In dataset.py: Add todo Comment

### 8/07 - jeff

- Fixed bugs in postprocessing about stroke2char
	1. Fix bugs in postprocessing about stroke2char
	2. Optimize some process
