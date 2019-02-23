# Csv Row Generator Function Definitions
# Michael Keim (michaelkeim2468@gmail.com)
import csv

# Generator function wrapper actual csv reader
# Will retrieve row values on the fly rather than storing in memory
def row_generator(file_name):
	with open(file_name) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			yield row

# Find number of enteries in each row
def row_size(file_name):
	with open(file_name) as csvfile:
		reader = csv.reader(csvfile)
		return len(next(reader))

# Find total number of rows
def col_size(file_name):
	with open(file_name) as csvfile:
		reader = csv.reader(csvfile)
		return len(list(reader))
