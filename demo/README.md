# Automatic processing for 6axis_revision Project
> 開發及維護人員: jefflin

> 最新更新時間: 2020/11/16

- [Automatic processing for 6axis_revision Project](#automatic-processing-for-6axis_revision-project)
	- [Quik Run](#quik-run)
	- [Arguments Setting](#arguments-setting)
			- [Example](#example)
	- [Program Running Rule](#program-running-rule)
		- [**注意事項**](#注意事項)
	- [Folder Structure](#folder-structure)

## Quik Run
- Run the following command for running postprocess without any arguments
	```
	python demo.py
	```

## Arguments Setting
See ```python demo.py -h``` output message.

#### Example
```
python demo.py --version 0
```

## Program Running Rule
自動化修正過程，包含以下三部分
1. 前處理
   - 輸入字元編號及誤差範圍，就可以生成數量僅一筆的測試資料
2. 修正
   - 修正六軸 
3. 後處理
   - 合併筆畫，及檢測輸出長度是否正確

### **注意事項**
- 請注意資料夾存放結構，否則容易報錯
- 輸出檔會照檔案類型分類到 `pic/`、`txt/`、`test_char/` 三個資料夾，其中 `test_char/` 存放Demo過程必需之資料，分別為以下四項
  - test_all_compare.png
  - test_all_input.txt
  - test_all_output.txt
  - test_all_target.txt

## Folder Structure

- `test.py`
- preprocessing/
- postprocessing/
- doc/
    - sample.yaml          ————————————————— doc
- demo/
    - `demo.py`
    - demo_util.py
    - dataset/             ——————————————————— root-path
        - 6axis/           ————————————————— input-path
            - char_00436_stroke.txt ...
        - test/             —————————————————— test-path
        - target/         ————————————————— target-path
    - logs/                  ———————————————————— logs-path
        - FSRCNN/
            - version_0/
                - FSRCNN_1x.pt
    - output/              ——————————————————— save-path
        - test_1_input.csv ...
        - pic/
            - test_1_compare.png ...
        - txt/
            - test_1_input.txt ...
        - test_char/
            - test_all_compare.png
            - test_all_input.txt
            - test_all_output.txt
            - test_all_target.txt
