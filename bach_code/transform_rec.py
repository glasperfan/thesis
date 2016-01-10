#########
#
## File: transform_rec.py
## Author: Hugh Zabriskie (c) 2016
## Description: Transform the chorale data into a format for training a recurrent neural network.
##				Each chorale should be a fixed-size vector (the length of the longest chorale, with
##				padding added for the others) where each element (index t) is an index corresponding 
##				to a set of features about the melody at that time t.
#
#########

from helpers import *
from collections import Counter
import h5py
import numpy as npy

# Print the number of unique lists in a list of lists
def uniqueLst(lst_of_lsts):
	st = set([tuple(i) for i in lst_of_lsts])
	return [list(i) for i in st]

# Flatten a list of lists
def flattenLst(lst_of_lsts):
	return [item for sublist in lst_of_lsts for item in sublist]

# Map the indices over the chorales
def mapIndices(X, idx_dict):
	X_mapped = []
	for chorale in X:
		Xmc = []
		for example in chorale:
			Xmc.append(idx_dict[tuple(example)])
		X_mapped.append(Xmc)
	return X_mapped

######################################################## 

# Load the original dataset
X_train = thawObject('X_train')
X_train_copy = thawObject('X_train')
y_train = thawObject('y_train')
X_test = thawObject('X_test')
X_test_copy = thawObject('X_test')
y_test = thawObject('y_test')
X_all = X_train + X_test
X_all_copy = X_train_copy + X_test_copy
y_all = y_train + y_test

# Set which features will be kept (the goal is to reduce the vocabulary size)
features = ['key', 'mode', 'time', 'beatstr', 'offset', 'cadence_dists', 'cadence?', 'pitch', 'pbefore', 'pafter']
features_to_delete = ['time', 'offset', 'cadence_dists', 'pafter', 'pbefore']

# flatten the chorales into a single dataset
# X = flattenLst(X)

# Create the dictionary mapping of indices to note features
for ftd in features_to_delete:
	if ftd not in features:
		raise Exception("%s not in the original feature set" % ftd)
	ftd_idx = features.index(ftd)
	for chorale in X_all:
		for example in chorale:
			del example[ftd_idx]
	del features[ftd_idx]

X_unique = uniqueLst(flattenLst(X_all))
X_dict = dict((tuple(v), k + 2) for k,v in enumerate(X_unique)) # 1-indexed for Torch

print "Reduced the vocabulary size from %d to %d." % (len(uniqueLst(flattenLst(X_all_copy))), len(X_unique))

# Pad the chorales
padding = [0,0,0,0]
X_dict[tuple(padding)] = 1
y_padding_idx = max(flattenLst(y_all)) + 1
print y_padding_idx
max_chorale_length = max(map(lambda lst: len(lst), X_all))

for idx, chorale in enumerate(X_train):
	padding_len = max_chorale_length - len(chorale)
	chorale += [padding] * padding_len
	y_train[idx] += [y_padding_idx] * padding_len

for idx, chorale in enumerate(X_test):
	padding_len = max_chorale_length - len(chorale)
	chorale += [padding] * padding_len
	y_test[idx] += [y_padding_idx] * padding_len


print "Chorales have been padding to a length of %d." % len(X_train[0])

# Perform mapping
X_train_rec = mapIndices(X_train, X_dict)
X_test_rec = mapIndices(X_test, X_dict)

# Testing (check dimensions)
assert len(X_train[0]) == len(y_train[0]) == len(X_test[0]) == len(y_test[0])
assert len(X_train_copy) == len(X_train_rec)
assert len(X_test_copy) == len(X_test_rec)

# Examine the first chorale
print "# First chorale below"
for i, lst in enumerate(X_train_copy[0]):
	print "%d\t -->\t %d" % (X_train_rec[0][i], y_train[0][i])
print "...plus %d units of padding." % (len(X_train_rec[0]) - len(X_train_copy[0]))

# Freeze the Python objects for future reference.
freezeObject(X_train_rec, "X_train_rec")
freezeObject(X_test_rec, "X_test_rec")
freezeObject(X_dict, "X_dict_rec")
freezeObject(y_train, "y_train_rec")
freezeObject(y_test, "y_test_rec")

# Write the data to the data/ folder.
for idx, score in enumerate(X_train_rec):
	with h5py.File("data/train_%d_rec.hdf5" % idx, "w", libver='latest') as f:
		X_vec = npy.array(X_train_rec[idx])
		f.create_dataset("X", X_vec.shape, dtype='i', data=X_vec)
		y_vec = npy.array(y_train[idx])
		f.create_dataset("y", y_vec.shape, dtype='i', data=y_vec)

for idx, score in enumerate(X_test_rec):
	with h5py.File("data/test_%d_rec.hdf5" % idx, "w", libver='latest') as f:
		X_vec = npy.array(X_test_rec[idx])
		f.create_dataset("X", X_vec.shape, dtype='i', data=X_vec)
		y_vec = npy.array(y_test[idx])
		f.create_dataset("y", y_vec.shape, dtype='i', data=y_vec)


with h5py.File("data/all_data_rec.hdf5", "w", libver='latest') as f:
	npyXtrain = npy.matrix(X_train_rec)
	f.create_dataset("X_train", npyXtrain.shape, dtype='i', data=npyXtrain)
	npyXtest = npy.matrix(X_test_rec)
	f.create_dataset("X_test", npyXtest.shape, dtype='i', data=npyXtest)
	npyytrain = npy.matrix(y_train)
	f.create_dataset("y_train", npyytrain.shape, dtype='i', data=npyytrain)
	npyytest = npy.matrix(y_test)
	f.create_dataset("y_test", npyytest.shape, dtype='i', data=npyytest)



