# Preprocess by Building Dataset

本資料夾為自動化建立書法的訓練資料，從機器手臂控制指令文字檔，按每個書法字及筆畫生成 target、train 及 test 三種 dataset。

> 開發人員: jefflin

- [Preprocess by Building Dataset](#preprocess-by-building-dataset)
  - [快速執行](#快速執行)
    - [生成 target 及 training data](#生成-target-及-training-data)
    - [生成 testing data](#生成-testing-data)
      - [Example](#example)
  - [軟體需求](#軟體需求)
  - [生成資料架構說明](#生成資料架構說明)
  - [參數說明](#參數說明)

---

## 快速執行

> 詳細執行方法及參數詳見 [參數說明](#參數說明)

<font color='red'>**注意:** 按照程式寫法，原本不能有相同名稱的資料夾存在 ( 預設資料夾為 `./dataset/` )</font>

### 生成 target 及 training data

`python main.py`

### 生成 testing data

`python main.py --test-char=TEST_CHAR`

#### Example

`python main.py --test-char=436`

---

## 軟體需求

- Python 3
- 以下為 Python 須安裝之套件
  - numpy
  - pandas
  - ArgumentParser
  - glob

---

## 生成資料架構說明

依照預設值，以下為產生的資料夾大致結構

> 資料夾及檔案名稱詳見 [參數說明](#參數說明)

- **dataset/**
  - **target/**
    - **0436/**
      - **01/**
        - 0436_01.csv
  - **train/**
    - **0436/**
      - **01/**
        - 0436_01_0001.csv
        - ...
        - 0436_01_0100.csv
  - **test/**
    - **0436/**
      - **01/**
        - 0436_01_0001.csv
        - ...
        - 0436_01_0030.csv

---

## 參數說明

- `-h, --help`

  - show this help message and exit

- `--train-num TRAIN_NUM`

  - 設定產生 training data 之 每一 stroke 的筆數 (預設值: 100)

- `--train-start-num TRAIN_START_NUM`

  - 設定產生之 training data ，編號要從哪裡開始 (預設值: 0)

- `--noise NOISE NOISE`

  - 設定雜點範圍 (預設值: [-1, 1])

- `--stroke-len STROKE_LEN`

  - 設定每一筆畫的長度 (預設值: 150)

- `--test-char TEST_CHAR`

  - 設定要產生 testing data 的書法字編號 (預設值: None)

- `--test-num TEST_NUM`

  - 設定產生 training data 之 每一 stroke 的筆數 (預設值: 30)

- `--char-idx CHAR_IDX`

  - 設定要用多少長度來儲存書法字編號 (以補 0 的方式) (預設值: 4)

- `--stroke-idx STROKE_IDX`

  - 設定要用多少長度來儲存筆畫編號 (以補 0 的方式) (預設值: 2)

- `--num-idx NUM_IDX`

  - 設定要用多少長度來儲存 data 筆數編號 (以補 0 的方式) (預設值: 4)

- `--input-path INPUT_PATH`

  - 設定讀取機器手臂控制指令之 TXT 檔資料夾路徑 (預設值: ./6axis/)

- `--root-path ROOT_PATH`

  - 設定輸出 data 的根資料夾 (預設值: ./dataset/)

- `--target-path TARGET_PATH`

  - 設定 target datas 資料夾路徑 (預設值: ./dataset/target/)

- `--train-path TRAIN_PATH`

  - 設定 training datas 資料夾路徑 (預設值: ./dataset/train/)

- `--test-path TEST_PATH`
  - 設定 testing datas 資料夾路徑 (預設值: ./dataset/test/)
