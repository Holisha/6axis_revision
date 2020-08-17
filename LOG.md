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

## Version 5

### 8/17 - fan

- Now model parameter and config would be save in log/\*/version\*/
	- `update`: In utils.py: `writer_builder` now would return writer and store path
		- return value: `writer` -> `(writer, model_path)`
	- `update`: In train.py: model_path would be define by `writer_builder`
	- `update`: In utils.py: `model_config` will save config to version directory
	- `update`: In train.py: progress bar can show current learning rate
	- `new`: In loss.py: add rmse loss function

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