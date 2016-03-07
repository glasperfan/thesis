#########
##
## File: oracle.py
## Author: Hugh Zabriskie (c) 2016
## Description: Preprocess the original data for an Oracle experiment so that the last feature of the
##				example at time t is the harmony chosen at time t. During training, that harmony is the
##				correct harmony that occurs on the previous beat.
##				Note: If the data point comes from the beginning of a chorale, this feature will be a
##				"padding" harmony.
##
#########

from helpers import *
import h5py
import numpy as npy


# Flatten a list of lists
def flattenLst(lst_of_lsts):
	return [item for sublist in lst_of_lsts for item in sublist]


# Load the original dataset
X_train = thawObject('X_train')
y_train = thawObject('y_train')
X_test = thawObject('X_test')
y_test = thawObject('y_test')
X_all = X_train + X_test
y_all = y_train + y_test

# Set "padding" harmony
max_y_idx = max(flattenLst(y_all))
max_x_idx = max(flattenLst(flattenLst(X_all)))
padding_idx = max_y_idx + 1

# Add oracle feature to X_train
for cidx, chorale in enumerate(X_train):
	for idx, example in enumerate(chorale):
		if idx == 0:
			example.append(padding_idx + max_x_idx)
		else:
			example.append(y_train[cidx][idx - 1] + max_x_idx)

print "# Sample chorale"
for idx, example in enumerate(X_train[0][:20]):
	print "%s \t %d \t (%d)" % (example, y_train[0][idx], y_train[0][idx] + max_x_idx)

# Freeze to verify
freezeObject(X_train, "X_train_oracle")

# Add blank feature to X_test (to be filled in during evaluation)
for chorale in X_test:
	for example in chorale:
		if idx == 0:
			example.append(1) # 1 will mark the beginning of a chorale
		else:
			example.append(2) # 2 for all others (this is just a convention)

with h5py.File("data/oracle.hdf5", "w", libver='latest') as f:
	npyXtrain = npy.matrix(flattenLst(X_train))
	f.create_dataset("X_train", npyXtrain.shape, dtype='i', data=npyXtrain)
	npyXtest = npy.matrix(flattenLst(X_test))
	f.create_dataset("X_test", npyXtest.shape, dtype='i', data=npyXtest)
	npyytrain = npy.array(flattenLst(y_train))
	f.create_dataset("y_train", npyytrain.shape, dtype='i', data=npyytrain)
	npyytest = npy.array(flattenLst(y_test))
	f.create_dataset("y_test", npyytest.shape, dtype='i', data=npyytest)















