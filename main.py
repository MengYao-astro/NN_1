# Program to Classify 16x16 Images of Digits
# Michael Keim (michaelkeim2468@gmail.com)
import numpy as np
import csv
import itertools
from sklearn.metrics import confusion_matrix, pairwise_distances

# Using a priori number of digits
digits = 10

# Read in data
data_in, data_out, test_data_in, test_data_out = np.genfromtxt('train_in.csv',delimiter=','), np.genfromtxt('train_out.csv',delimiter=',', dtype=int), np.genfromtxt('test_in.csv',delimiter=','), np.genfromtxt('test_out.csv',delimiter=',', dtype=int)
train_points, imsize, test_points = len(data_in), len(data_in[0]), len(test_data_in)

# Read in training set digits while calculating center and digit totals
center, number = np.zeros((digits, imsize)), np.zeros(digits)
for i in range(train_points):
		number[data_out[i]], center[data_out[i]] = number[data_out[i]]+1., center[data_out[i]]+data_in[i,:]
center /= number[:, None]

# Find maximum radii
rtemp2, rmax2 = np.zeros(imsize), np.zeros((digits, imsize))
for i in range(train_points):
	if rtemp2.sum() > rmax2[data_out[i]].sum():
		rmax2[data_out[i]] = rtemp2

# Calculate distances between cloud centers and find the closest digits
distance, minij = np.zeros((digits, digits)), [0, 1]
for i in range(digits):
	for j in range(i):
		distance[i, j] = ((np.square(center[i, :]-center[j, :])).sum())**(0.5)
		distance[j, i] = distance[i, j]
		if distance[i, j] < distance[minij[0], minij[1]]:
			minij = [i, j]
	print("Cloud " + str(i) + " has " +str(number[i]) + " points and a maximum distance from center of " + str((rmax2[i].sum())**(0.5)))
print("The distance between clouds is: \n", distance.round(3), "\nThe closest digits are " + str(minij[0]) + " and " + str(minij[1]))

# Now classify based on distances
dtemp, training_class_euc, train_percent_euc = np.zeros(digits), np.zeros(train_points, dtype=int), 0.0
for k in range(train_points):
	for i in range(digits):
		dtemp[i] = (pairwise_distances(np.vstack((data_in[k,:], center[i,:])), metric='euclidean'))[0,1]
	training_class_euc[k] = np.argmin(dtemp)
	if training_class_euc[k] == data_out[k]:
		train_percent_euc += 1.0
	dtemp *= 0.0
print(str(100.*train_percent_euc/train_points) + " percent correctly classified\nConfusion Matrix from Euclidean Distances for Training Set:\n", confusion_matrix(data_out, training_class_euc))

# Now classify based on distances
testing_class_euc, test_percent_euc = np.zeros(test_points, dtype=int), 0.0
for k in range(test_points):
	for i in range(digits):
			dtemp[i] = (pairwise_distances(np.vstack((test_data_in[k,:], center[i,:])), metric='euclidean'))[0,1]
	testing_class_euc[k] = np.argmin(dtemp)
	if testing_class_euc[k] == test_data_out[k]:
		test_percent_euc += 1.0
	dtemp *= 0.0
print(str(100.*test_percent_euc/test_points) + " percent correctly classified\nConfusion Matrix from Euclidean Distances for Training Set:\n", confusion_matrix(test_data_out, testing_class_euc))