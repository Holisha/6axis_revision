# Postprocessing for 6axis_revision Project
> 開發及維護人員: jefflin

> 最新更新時間: 2020/08/05 02:10

## Quik Run
Run the following command for running without any arguments
```
python postprocessor.py
```

## Arguments
Set the path of the directory. (default : `./output` )
```
--path PATH
```

## Program Running Rule
採遞迴方式取出預設資料夾底下所有層的 csv 檔，完成以下三個功能:
1. stroke2char: 把 `test_` 開頭的 csv 檔，依筆畫順序合併成單一完整書法字的 csv 檔，輸出檔名為 `test_all_(target|input|output).csv`
2. axis2img: 把 target, input, output 三部分生成 2D 細線化圖示，以進行比較
3. csv2txt: 把所有 csv 檔轉成機器手臂可執行的 txt 指令檔

注意:
- test data 的檔名須為 `test_` 開頭
- 檔名須以 `_target.csv`, `_input.csv` 或 `_output.csv` 為結尾形式命名
- 因此，同一層資料夾下的 csv 檔個數須為 3 的倍數，不然會報錯