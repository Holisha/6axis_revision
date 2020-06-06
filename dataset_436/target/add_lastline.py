import os
import csv

number =5
tol_len=150

for i in range(1,number+1):
    with open(str(i)+'.csv', newline='', encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile)
        print(f'rows={rows}')
        row_count=0
        for row in rows:
            words = row
            row_count+=1
        addlines=tol_len-row_count
    with open(str(i)+'.csv', 'a',newline='') as csvfile:
        for item in range(addlines):
            writer = csv.writer(csvfile)
            writer.writerow(words)
