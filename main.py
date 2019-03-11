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
sklrn =  ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
dtemp, train_class, test_class, percent = np.zeros(digits), np.zeros(train_points, dtype=int), np.zeros(test_points, dtype=int), 0.0
for s in sklrn:
	for k in range(train_points):
		for i in range(digits):
			dtemp[i] = (pairwise_distances(np.vstack((data_in[k,:], center[i,:])), metric=s))[0,1]
		train_class[k] = np.argmin(dtemp)
		if train_class[k] == data_out[k]:
			percent += 1.0
		dtemp *= 0.0
	print(str(100.*percent/train_points) + " percent of training set correctly classified for " + s + " distance metric\nConfusion matrix for training set:\n", confusion_matrix(data_out, train_class))
	train_class *= 0
	percent *= 0.0

	# Now for the test set
	for k in range(test_points):
		for i in range(digits):
			dtemp[i] = (pairwise_distances(np.vstack((test_data_in[k,:], center[i,:])), metric=s))[0,1]
		test_class[k] = np.argmin(dtemp)
		if test_class[k] == test_data_out[k]:
			percent += 1.0
		dtemp *= 0.0
	print(str(100.*percent/test_points) + " percent of test set correctly classified for " + s + " distance metric\nConfusion matrix for test set:\n", confusion_matrix(test_data_out, test_class))
	test_class *= 0
	percent *= 0.0