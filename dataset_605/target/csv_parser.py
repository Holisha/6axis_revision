import os, csv, re

def main(orin_csv_name='../char00605_stroke.csv'):
	"""Parser oringinal whole character csvfile to each stroke.

	Keyword Arguments:
		orin_csv_name {str} -- oringin csv file (default: {'../char00605_stroke.csv'})
	"""
	with open(orin_csv_name, newline='') as orin_csv_file:
		orin_rows = csv.reader(orin_csv_file)
		for orin_row in orin_rows:
			stroke_num = re.search(r'\d+', orin_row[-1]).group()
			with open(f'{stroke_num}.csv', 'a', newline='') as stroke_csv_file:
				writer = csv.writer(stroke_csv_file)
				writer.writerow(orin_row)

if __name__ == '__main__':
	main()