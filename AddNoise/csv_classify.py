import os,shutil
import csv
from add_lastline import add
# number = int(input("How much pic do you want: "))
# tol_strokes = int(input("How many strokes does the word contained: "))
def classify(number, tol_strokes):
    # number=1000
    # tol_strokes=28
    print(os.getcwd())
    os.chdir("6axis_data")
    try:
        for i in range(1,tol_strokes+1):
            os.mkdir(str(i))       
    except:
        pass

    flag=0
    cnt=0
    for i in range(1,number+1):
    # 開啟 CSV 檔案
        with open('6axis_data'+str(i)+'.csv', newline='') as csvfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)
            past=""
            # 以迴圈輸出每一列
            for row in rows:
                idx=row[6][6:]
                if idx!=past:
                    past=idx
                    csvfile=open(str(idx)+'_'+str(i)+'.csv', 'w', newline='') 
                    # 建立 CSV 檔寫入器
                    writer = csv.writer(csvfile)
                # 寫入一列資料
                writer.writerow(row)    
        csvfile.close()

    for i in range(1,number+1):
        for j in range(1,tol_strokes+1):
            shutil.move(str(j)+"_"+str(i)+".csv",str(j))
    os.chdir("../..")
    add(number,tol_strokes)
