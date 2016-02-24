
#
# c_rnn.py
# Restructure target data for RNN approach.
#

import numpy as np
import h5py
from helpers import freezeObject
np.set_printoptions(threshold=np.nan)

DATA_FILE = "data/chorales.hdf5"
OUT_FILE = "data/chorales_rnn.hdf5"
MAX_SEQ_LEN = 0

def load_dataset(name):
	dataX, datay = [], []
	with h5py.File(DATA_FILE, "r", libver='latest') as f:
		counter = 0
		while True:
			try:
				dataX.append(f['%s/chorale%d_X' % (name, counter)].value)
				datay.append(f['%s/chorale%d_y' % (name, counter)].value)
			except:
				break
			counter += 1
	return dataX, datay

# Individual chorales
trainXc, trainyc = load_dataset("train")
devXc, devyc = load_dataset("dev")
testXc, testyc = load_dataset("test")

# Aggregated datasets
with h5py.File(DATA_FILE, "r", libver='latest') as f:
	ytrain = f['ytrain'].value
	ydev = f['ydev'].value
	ytest = f['ytest'].value
	yall = np.vstack((ytrain, ydev, ytest))
	Xtrain = f['Xtrain'].value
	Xdev = f['Xdev'].value
	Xtest = f['Xtest'].value
	Xall = np.vstack((Xtrain, Xdev, Xtest))

# Create the map between features and indices
harmdict = {}
for idx, feature in enumerate(list(set(map(tuple, yall)))):
	harmdict[feature] = idx + 1
f = lambda x: harmdict[tuple(x)]
yall_rnn = np.array(map(f, yall))

# Add padding to X
xpadding = [min(Xall[:, i + 1]) - 1 for i in range(10)] + [max(Xall[:, i]) for i in range(10, 13)]
ypadding = yall_rnn.max()
MAX_SEQ_LEN = max([x.shape[0] for x in trainXc + devXc + testXc])


def pad_one(chorale, padding, stack):
	global f
	if type(padding) != list:
		chorale = map(f, chorale)
	if MAX_SEQ_LEN - len(chorale) > 0:
		p = [padding] * (MAX_SEQ_LEN - len(chorale))
		try:
			chorale = stack((chorale, np.array(p)))
		except:
			print chorale.shape, np.array(p).shape
	assert len(chorale) == MAX_SEQ_LEN
	return np.array(chorale)

def pad(dataset, padding):
	stack = np.vstack if type(padding) == list else np.hstack
	for idx in range(len(dataset)):
		dataset[idx] = pad_one(dataset[idx], padding, stack)
	return dataset

trainXc, devXc, testXc = pad(trainXc, xpadding), pad(devXc, xpadding), pad(testXc, xpadding)
trainyc, devyc, testyc = pad(trainyc, ypadding), pad(devyc, ypadding), pad(testyc, ypadding)

freezeObject(trainXc[0], "trainXc0")
freezeObject(trainyc[0], "trainyc0")
freezeObject(harmdict, "harmdict")

# Apply and write
with h5py.File(OUT_FILE, "w", libver='latest') as fl:
	# ytrain, ydev, ytest, yall
	for name, data in [("ytrain", ytrain), ("ydev", ydev), ("ytest", ytest), ("yall", yall)]:
		out = np.array(map(f, data))
		fl.create_dataset(name, out.shape, dtype="i", data=out)
	# training
	for idx, (X, yc) in enumerate(zip(trainXc, trainyc)):
		fl.create_dataset("train/chorale%d_X" % idx, X.shape, dtype="i", data=X)
		fl.create_dataset("train/chorale%d_y" % idx, yc.shape, dtype="i", data=yc)
	# dev
	for idx, (X, y) in enumerate(zip(devXc, devyc)):
		fl.create_dataset("dev/chorale%d_X" % idx, X.shape, dtype="i", data=X)
		fl.create_dataset("dev/chorale%d_y" % idx, y.shape, dtype="i", data=y)
	# tests
	for idx, (X, y) in enumerate(zip(testXc, testyc)):
		fl.create_dataset("test/chorale%d_X" % idx, X.shape, dtype="i", data=X)
		fl.create_dataset("test/chorale%d_y" % idx, y.shape, dtype="i", data=y)
	metadata = np.array([ypadding, max(xpadding), len(trainXc[0])])
	fl.create_dataset("metadata", metadata.shape, dtype="i", data=metadata)