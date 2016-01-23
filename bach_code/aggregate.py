#########
##
## File: aggregate.py
## Author: Hugh Zabriskie (c) 2016
## Description: Aggregate the chorale data files into a single HDF5 data file to facilitate use with Torch7. 
##
## TODO: make this happen in featurize.py
##
#########

import glob
import h5py
import numpy
from helpers import freezeObject

numpy.set_printoptions(threshold='nan')

INPUT_DIR = "data/"
OUTPUT_DIR = "data/"

Xtrain, ytrain, Xtest, ytest = [], [], [], []

for fn in glob.glob(INPUT_DIR + 'train_*.hdf5'):
	with h5py.File(fn, "r", libver='latest') as f:
		Xtrain.append(f['X'].value)
		ytrain.append(f['y'].value)

for fn in glob.glob(INPUT_DIR + 'test_*.hdf5'):
	with h5py.File(fn, "r", libver='latest') as f:
		Xtest.append(f['X'].value)
		ytest.append(f['y'].value)

# Create a binary vector for ytest that signals where chorales begin and end
# 1 signals the beginning of a chorale
def add_signal_col(chorale):
	col = numpy.zeros(chorale.shape[0])
	col[0] = 1
	return col

signal_col = map(add_signal_col, ytest)

Xtrain = reduce(lambda x1, x2: numpy.vstack((x1, x2)), Xtrain)
ytrain = reduce(lambda y1, y2: numpy.vstack((y1, y2)), ytrain)
Xtest = reduce(lambda x1, x2: numpy.vstack((x1, x2)), Xtest)
ytest = reduce(lambda y1, y2: numpy.vstack((y1, y2)), ytest)
signal_col = reduce(lambda n1, n2: numpy.hstack((n1, n2)), signal_col)

print "Xtrain: " + str(Xtrain.shape)
print "ytrain: " + str(ytrain.shape)
print "Xtest: " + str(Xtest.shape)
print "ytest: " + str(ytest.shape)
print "signal col: " + str(signal_col.shape)

freezeObject(Xtrain, "Xtrain_agg")
freezeObject(ytrain, "ytrain_agg")
freezeObject(Xtest, "Xtest_agg")
freezeObject(ytest, "ytest_agg")
freezeObject(signal_col, "signal_col_agg")

with h5py.File(OUTPUT_DIR + "all_data.hdf5", "w", libver='latest') as f:
	f.create_dataset("X_train", Xtrain.shape, dtype='i', data=Xtrain)
	f.create_dataset("y_train", ytrain.shape, dtype='i', data=ytrain)
	f.create_dataset("X_test", Xtest.shape, dtype='i', data=Xtest)
	f.create_dataset("y_test", ytest.shape, dtype='i', data=ytest)
	f.create_dataset("test_signal_col", signal_col.shape, dtype='i', data=signal_col)
