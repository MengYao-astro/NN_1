# Program to Classify 16x16 Images of Digits
# Michael Keim (michaelkeim2468@gmail.com)
import numpy as np
import csv
from sklearn.metrics import confusion_matrix 

# Specify training and test sets
train_in, train_out, test_in, test_out = "train_in.csv", "train_out.csv", "test_in.csv", "test_out.csv"

# Using a priori number of digits
digits = 10

# Find image size and number of points
with open(train_in) as csvfile:
	reader = csv.reader(csvfile)
	imsize, points = len(next(reader)), 1+len(list(reader))
with open(test_in) as csvfile:
	reader = csv.reader(csvfile)
	test_points = len(list(reader))

# Create matricies for data
data_in, data_out = np.zeros((points, imsize)), np.zeros(points, dtype=int)
test_data_in, test_data_out = np.zeros((test_points, imsize)), np.zeros(test_points, dtype=int)
center, number = np.zeros((digits, imsize)), np.zeros(digits)

# Read in training set while calculating center and digit totals
with open(train_in) as csvfile:
	reader = csv.reader(csvfile)
	i = 0
	for row in reader:
		data_in[i] = row
		i += 1
with open(train_out) as csvfile:
	reader = csv.reader(csvfile)
	i = 0
	for row in reader:
		data_out[i] = row[0]
		number[data_out[i]] += 1.0
		center[data_out[i]] += data_in[i]
		i += 1
for i in range(digits):
	center[i,:] /= number[i]

# Find maximum radii
rtemp2, rmax2 = np.zeros(imsize), np.zeros((digits, imsize))
for i in range(points):

	# Find distance^2 from center for this image
	for j in range(imsize):
		rtemp2[j] = (data_in[i, j]-center[data_out[i], j])**2.0

	# If distance^2 is greater than that stored for a given digit, replace
	if rtemp2.sum() > rmax2[data_out[i]].sum():
		rmax2[data_out[i]] = rtemp2

# Calculate distances between cloud centers and find the closest digits
distance = np.zeros((digits, digits))
minij = [0, 1]
for i in range(digits):
	for j in range(i):
		for k in range(imsize):
			distance[i, j] += (center[i, k]-center[j, k])**2.0
		distance[i, j] = (distance[i, j])**(0.5)
		distance[j, i] = distance[i, j]
		if distance[i, j] < distance[minij[0], minij[1]]:
			minij = [i, j]

# Now classify based on distances
dtemp = np.zeros(digits)
class_id = np.zeros(points, dtype=int)
for k in range(points):
	for i in range(digits):
		for j in range(imsize):
			dtemp[i] += (data_in[k,j]-center[i,j])**2.0
	class_id[k] = np.argmin(dtemp)
	dtemp *= 0.0
print(confusion_matrix(data_out, class_id))

"""
# Read in test set
with open(test_in) as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		test_data_in[i] = row
		i += 1
with open(train_out) as csvfile:
	reader = csv.reader(csvfile)
	i = 0
	for row in reader:
		data_out[i] = row[0]
		number[data_out[i]] += 1.0
		center[data_out[i]] += data_in[i]
		i += 1
for i in range(digits):
	center[i,:] /= number[i]
	"""


