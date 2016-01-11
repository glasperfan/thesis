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

Xtrain = reduce(lambda x1, x2: numpy.vstack((x1, x2)), Xtrain)
ytrain = reduce(lambda y1, y2: numpy.vstack((y1, y2)), ytrain)
Xtest = reduce(lambda x1, x2: numpy.vstack((x1, x2)), Xtest)
ytest = reduce(lambda y1, y2: numpy.vstack((y1, y2)), ytest)

print Xtrain.shape
print ytrain.shape
print ytrain[:5]
print Xtest.shape
print ytest.shape

with h5py.File(OUTPUT_DIR + "all_data.hdf5", "w", libver='latest') as f:
	f.create_dataset("X_train", Xtrain.shape, dtype='i', data=Xtrain)
	f.create_dataset("y_train", ytrain.shape, dtype='i', data=ytrain)
	f.create_dataset("X_test", Xtest.shape, dtype='i', data=Xtest)
	f.create_dataset("y_test", ytest.shape, dtype='i', data=ytest)
