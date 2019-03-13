# Program to Classify 16x16 Images of Digits
# Following report guidlines author names and emails are excluded
import numpy as np
import csv
import itertools
from sklearn.metrics import confusion_matrix, pairwise_distances

# Function to calculate confusion matricies from minimum center distance or perceptron weights
def confusion(inputs, outputs, labels, ftype, **kwargs):
	temp, classification, percent = np.zeros(labels), np.zeros(len(inputs), dtype=int), 0.
	metric, centers , wieght = kwargs.get('skltype', None), kwargs.get('centers', None), kwargs.get('weight', None)
	for i in range(len(inputs)):
		if ftype == "distance":
			classification[i] = np.argmin([(pairwise_distances(np.vstack((inputs[i,:], centers[j,:])), metric=metric))[0,1] for j in range(digits)])
			percent = percent+1. if classification[i] == outputs[i] else percent
		elif ftype == "perceptron":
			classification[i] = np.argmax([np.dot(inputs[i,:], weights[j,:]) for j in range(digits)])
			percent = percent+1. if classification[i] == outputs[i] else percent
	print(str(round(100.*percent/len(inputs),3)) + " percent correctly classified.\nConfusion matrix:\n", confusion_matrix(outputs, classification))

# Read in data
data_in, data_out, test_data_in, test_data_out = np.genfromtxt('train_in.csv',delimiter=','), np.genfromtxt('train_out.csv',delimiter=',', dtype=int), np.genfromtxt('test_in.csv',delimiter=','), np.genfromtxt('test_out.csv',delimiter=',', dtype=int)
train_points, imsize, test_points, digits = len(data_in), len(data_in[0]), len(test_data_in), len(np.unique(data_out))

# Read in training set digits while calculating center and digit totals
center, number = np.zeros((digits, imsize)), np.zeros(digits)
for i in range(train_points):
		number[data_out[i]], center[data_out[i]] = number[data_out[i]]+1., center[data_out[i]]+data_in[i,:]
center /= number[:, None]

# Find maximum radii
rtemp2, rmax2 = np.zeros(imsize), np.zeros((digits, imsize))
for i in range(train_points):
	rtemp2 = (data_in[i, :]-center[data_out[i], :])**2.0
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
	print("Cloud " + str(i) + " has " +str(int(number[i])) + " points and a maximum distance from center of " + str(((rmax2[i].sum())**(0.5)).round(3)))
print("\nThe distance between clouds is: \n", distance.round(3), "\nThe closest digits are " + str(minij[0]) + " and " + str(minij[1]))

# Now classify based on distances
sklrn =  ['cosine', 'euclidean', 'manhattan']
for s in sklrn:
	print("\nClassifying training set by " + s + " distance metric ...\n")
	confusion(data_in, data_out, digits, ftype='distance', skltype=s, centers=center)
	print("\nClassifying test set by " + s + " distance metric ...\n")
	confusion(test_data_in, test_data_out, digits, ftype='distance', skltype=s, centers=center)

# Append bias and make initial weights
train_perceptron, test_perceptron, weights = np.append(np.ones((train_points,1)), data_in, axis=1), np.append(np.ones((test_points,1)), test_data_in, axis=1), np.zeros((digits, imsize+1))

# Specify perceptron properties
learning_rate, max_iter, ident, predict = 0.01, 100, 0, 0

# Train perceptron with heaviside disciminant function
print("\nTraining perceptron ...")
for _ in range(max_iter):
	for image, digit in zip(train_perceptron, data_out):
		for i in range(digits):
			ident = 1 if digit == i else 0
			predict = 1 if np.dot(image, weights[i,:]) > 0. else 0
			weights[i,:] += learning_rate * (ident - predict) * image
print("\nClassifying training set with perceptron ...\n")
confusion(train_perceptron, data_out, digits, ftype='perceptron', weight=weights)
print("\nClassifying test set with perceptron ...\n")
confusion(test_perceptron, test_data_out, digits, ftype='perceptron', weight=weights)