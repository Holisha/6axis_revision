import os
import csv

def main(number=14, tol_len=150):
    """Add last line of csvfile for same stroke length.

    Keyword Arguments:
        number {int} -- stroke number (default: {14})
        tol_len {int} -- each stroke length (default: {150})
    """
    for i in range(1,number+1):
        with open(str(i)+'.csv', newline='', encoding="utf-8") as csvfile:
            rows = csv.reader(csvfile)
            row_count=0
            for row in rows:
                words = row
                row_count+=1
            addlines=tol_len-row_count
        with open(str(i)+'.csv', 'a',newline='') as csvfile:
            for item in range(addlines):
                writer = csv.writer(csvfile)
                writer.writerow(words)

if __name__ == '__main__':
	main()