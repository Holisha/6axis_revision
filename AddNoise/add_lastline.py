import os,shutil
import csv

def add(number,tol_strokes):
    # print(os.getcwd())
    tol_len=150

    for i in range(1,number+1):
        for j in range(1,tol_strokes+1):
            with open('new6axis/6axis_data/'+str(j)+'/'+str(j)+"_"+str(i)+".csv", newline='') as csvfile:
                rows = csv.reader(csvfile)
                row_count=0
                for row in rows:
                    words = row
                    row_count+=1
                addlines=tol_len-row_count
            with open('new6axis/6axis_data/'+str(j)+'/'+str(j)+"_"+str(i)+".csv", 'a',newline='') as csvfile:
                for item in range(addlines):
                    writer = csv.writer(csvfile)
                    writer.writerow(words)
