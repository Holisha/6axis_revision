import csv
import os
from glob import glob

def main(path='./output'):
	"""Convert all CSV files to TXT files.

	Keyword Arguments:
		path {str} -- Path to output directory (default: {'./output'})
	"""
	for csv_name in sorted(glob(os.path.join(path, '*.csv'))):
		# read csv file content
		with open(csv_name, newline='') as csv_file:
			rows = csv.reader(csv_file)
			txt_name = f'{csv_name[:-4]}.txt'
			# store in txt file
			with open(txt_name, "w") as txt_file:
				for row in rows:
					txt_file.write("movl 0 ")
					for j in range(len(row) - 1):
						txt_file.write(f'{float(row[j]):0.4f} ')
					txt_file.write("100.0000 ")
					txt_file.write(f'{row[6]}\n')

if __name__ == '__main__':
	main()