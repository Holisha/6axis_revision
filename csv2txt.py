import csv
import os

def main(path='./output'):
	"""Convert all CSV files to TXT files.

	Keyword Arguments:
		path {str} -- Path to output directory (default: {'./output'})
	"""
	for root, dirs, files in os.walk(path):
		for f in files:
			# if not csv file
			if (f[-4:] != '.csv'):
				continue;
			
			csv_name = os.path.join(root, f);
			print(f'read\t{csv_name}')
			
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
					print(f'write\t{txt_name}')

if __name__ == '__main__':
	main()