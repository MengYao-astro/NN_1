# Program to Classify 16x16 Images of Digits
# Michael Keim (michaelkeim2468@gmail.com)
import numpy as np
import csv
import itertools
from sklearn.metrics import confusion_matrix 

# Specify training and test sets
train_in, train_out, test_in, test_out = "train_in.csv", "train_out.csv", "test_in.csv", "test_out.csv"

# Using a priori number of digits
digits = 10

# Read in training data
with open(train_in) as csvfile:
	points, size, data = itertools.tee(csv.reader(csvfile), 3)

	# Find image size and number of points
	train_points, imsize = len(list(points)), len(next(size))

	# Create matricies for data
	data_in, data_out = np.zeros((train_points, imsize)), np.zeros(train_points, dtype=int)
	center, number = np.zeros((digits, imsize)), np.zeros(digits)

	# Read in training set images
	i = 0
	for row in data:
		data_in[i] = row
		i += 1

# Read in training set digits while calculating center and digit totals
with open(train_out) as csvfile:
	train_digits = csv.reader(csvfile)
	i = 0
	for row in train_digits:
		data_out[i] = row[0]
		number[data_out[i]] += 1.0
		center[data_out[i]] += data_in[i]
		i += 1
for i in range(digits):
	center[i,:] /= number[i]

# Find maximum radii
rtemp2, rmax2 = np.zeros(imsize), np.zeros((digits, imsize))
for i in range(train_points):

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
	print("Cloud " + str(i) + " has " +str(number[i]) + " points and a maximum distance from center of " + str((rmax2[i].sum())**(0.5)))
print("The distance between clouds is:")
print(distance.round(3))
print("The closest digits are " + str(minij[0]) + " and " + str(minij[1]))

# Now classify based on distances
dtemp = np.zeros(digits)
training_class_euc = np.zeros(train_points, dtype=int)
train_percent_euc = 0.0
for k in range(train_points):
	for i in range(digits):
		for j in range(imsize):
			dtemp[i] += (data_in[k,j]-center[i,j])**2.0
	training_class_euc[k] = np.argmin(dtemp)
	if training_class_euc[k] == data_out[k]:
		train_percent_euc += 1.0
	dtemp *= 0.0
print(str(100.*train_percent_euc/train_points) + " percent correctly classified")
print("Confusion Matrix from Euclidean Distances for Training Set:")
print(confusion_matrix(data_out, training_class_euc))

# Read in testing data
with open(test_in) as csvfile:
	test_size, test_data = itertools.tee(csv.reader(csvfile), 2)

	# Find image size and number of train_points
	test_points = len(list(test_size))

	# Create matricies for data
	test_data_in, test_data_out = np.zeros((test_points, imsize)), np.zeros(test_points, dtype=int)

	# Read in training set images
	i = 0
	for row in test_data:
		test_data_in[i] = row
		i += 1

# Read in test set digits
with open(test_out) as csvfile:
	test_digits = csv.reader(csvfile)
	i = 0
	for row in test_digits:
		test_data_out[i] = row[0]
		i += 1

# Now classify based on distances
testing_class_euc = np.zeros(test_points, dtype=int)
test_percent_euc = 0.0
for k in range(test_points):
	for i in range(digits):
		for j in range(imsize):
			dtemp[i] += (test_data_in[k,j]-center[i,j])**2.0
	testing_class_euc[k] = np.argmin(dtemp)
	if testing_class_euc[k] == test_data_out[k]:
		test_percent_euc += 1.0
	dtemp *= 0.0
print(str(100.*test_percent_euc/test_points) + " percent correctly classified")
print("Confusion Matrix from Euclidean Distances for Training Set:")
print(confusion_matrix(test_data_out, testing_class_euc))