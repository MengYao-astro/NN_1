# Program to Classify 16x16 Images of Digits
# Michael Keim (michaelkeim2468@gmail.com)
import numpy as np
import data_reader as datain

# Specify training set
train_in, train_out = "data/train_in.csv", "data/train_out.csv"

# Using a priori number of digits
digits, imsize, points = 10, datain.row_size(train_in), datain.col_size(train_in)
center, number = np.zeros((digits, imsize)), np.zeros(digits)

# To avoid possible memory issues, data not stored in memory
# First loop over rows to determine center and number of points per digit
# Note that zip returns an iterator as of Python 3
for row0, row1 in zip(datain.row_generator(train_in), datain.row_generator(train_out)):

	# Add up sum for calculating mean
	for i in range(imsize):
		center[int(row1[0]), i] += float(row0[i])

	# Count Digits
	number[int(row1[0])] += 1.0

# Divide by N to find center
for i in range(digits):
	center[i,:] /= number[i]

# Second loop over rows to determine maximum radii
rmaxval2, rtemp, rmax = np.zeros(digits), np.zeros((digits, imsize)), np.zeros((digits, imsize))
for row0, row1 in zip(datain.row_generator(train_in), datain.row_generator(train_out)):

	# Find distance^2 from center for this image
	for i in range(imsize):
		rtemp[int(row1[0]), i] = (float(row0[i])-center[int(row1[0]), i])**2.0

	# If distance^2 is greater than that stored for a given digit, replace
	if rtemp.sum() > rmaxval2[int(row1[0])]:
		rmaxval2[int(row1[0])] = rtemp.sum()
		for i in range(imsize):
			rmax[int(row1[0]), i] = rtemp[int(row1[0]), i]

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
dclass = np.zeros(points, dtype=int)
n = 0
for row0 in datain.row_generator(train_in):
	for i in range(digits):
		for j in range(imsize):
			dtemp[i] += (float(row0[j])-center[i,j])**2.0
	dclass[n] = np.argmin(dtemp)
	dtemp *= 0.0
	n += 1



