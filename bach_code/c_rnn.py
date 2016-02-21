
#
# c_rnn.py
# Restructure data for RNN approach.
#

from rnn import load
import numpy as np
import h5py

cX, cy, cXall, cyall = load()

# Create the map between features and indices
harmdict = {}
for idx, feature in enumerate(list(set(map(tuple, cyall)))):
	harmdict[feature] = idx + 1

cy_full = []
f = lambda x: harmdict[tuple(x)]
with h5py.File('data/chorales_rnn.hdf5', "w", libver='latest') as fl:
	for idx, chorale in enumerate(cy):
		data = np.array(map(f, chorale))
		fl.create_dataset('chorale%d_y' % idx, data.shape, dtype="i", data=data)
	data_all = np.array(map(f, cyall))
	fl.create_dataset('yall', data_all.shape, dtype="i", data=data_all)