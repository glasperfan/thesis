import h5py
import numpy

# For nn.Sequencer, both input and output are in the form `seqlen * batchsize * featsize`.

# A chorale is a defined as a "batch."

# **`seqlen`** = number of time steps of longest phrase (input/output)

# **`batchsize`** = number of phrases in the chorale

# **`feats`** = number of features per time step

# Constants
CAD_COL = 6
CAD_ON = 161
CAD_OFF = 160
SEQLEN = 0

def load():
	cX, cy = [], []
	with h5py.File('data/chorales.hdf5', "r", libver='latest') as f:
		Xtrain = f['Xtrain'].value
		ytrain = f['ytrain'].value
		Xdev = f['Xdev'].value
		ydev = f['ydev'].value
		Xtest = f['Xtest'].value
		ytest = f['ytest'].value
		cXall = numpy.vstack((Xtrain, Xdev, Xtest))
		cyall = numpy.vstack((ytrain, ydev, ytest))
		counter = 0
		while True:
			try:
				cX.append(f['chorale%d_X' % counter].value)
				cy.append(f['chorale%d_y' % counter].value)
			except:
				break
			counter += 1
	print "%d chorales collected" % counter
	return cX, cy, cXall, cyall
    

# [1 1 3 4] --> [1 13 26 42] or something like that
def target_update(cy, cols_max):


# Generate a batch out of a chorale
# cX = chorale in the shape len * featsize
# cy = harmonization in the shape seqlen * featsize
# Outputs (hX, hy), each of which are in the form seqlen * batchsize * featsize
# seqlen = length of LONGEST possible phrase
# batchsize = number of phrases in the chorale
# featsize = same as input featsize
# 1 batch = 1 chorale
# 1 sequence is one phrase, where each phrase ends with a fermata
def batch(cX, cy):
	global SEQLEN
	is_cadence_col = 6
	# Batch
	batchX, batchy, cbX, cby = [], [], [], []
	for i in range(len(cX)):
		end_of_phrase = cX[i-1][is_cadence_col] == CAD_ON and cX[i][is_cadence_col] == CAD_OFF
		if end_of_phrase and i > 0:
			batchX.append(numpy.matrix(cbX))
			batchy.append(numpy.matrix(cby))
			SEQLEN = max(SEQLEN, len(cbX))
			cbX, cby = [], []
		cbX.append(cX[i])
		cby.append(cy[i])
	batchX.append(numpy.matrix(cbX))
	batchy.append(numpy.matrix(cby))
	SEQLEN = max(SEQLEN, len(cbX))
	assert cbX[-1][is_cadence_col] == CAD_ON # basic checks
	assert numpy.array_equal(cX[-1], cbX[-1])
	return batchX, batchy


def pad(cX, cy, max_seq_len, xpadding, ypadding):
	for idx in range(len(cX)):
		if max_seq_len - len(cX[idx]) > 0:
			padding = [xpadding] * (max_seq_len - len(cX[idx]))
			cX[idx] = numpy.vstack((cX[idx], numpy.matrix(padding)))
	for idx in range(len(cy)):
		if max_seq_len - len(cy[idx]) > 0:
			padding = [ypadding] * (max_seq_len - len(cy[idx]))
			cy[idx] = numpy.vstack((cy[idx], numpy.matrix(padding)))
	assert len(cX[0]) == max_seq_len
	assert len(cy[0]) == max_seq_len
	return cX, cy



# Load
cX, cy, cXall, cyall = load()

# Update targets to be one-hot vectors
max_target_idx = [max(cyall[:, i]) for i in range(len(cyall[0]))]
for i in range(len(cy)):
	cy[i] = target_update(cy[i], max_target_idx)
print "target vectors updated..."

# Batch
for i in range(len(cX)):
	cX[i], cy[i] = batch(cX[i], cy[i])
print "batched..."

# Pad
xpadding = [max(cXall[:, i]) + 1 for i in range(len(cXall[0]))]
ypadding = [max(cyall[:, i]) + 1 for i in range(len(cyall[0]))]
for i in range(len(cX)):
	cX[i], cy[i] = pad(cX[i], cy[i], SEQLEN, xpadding, ypadding)
print "padded..."

# Write
with h5py.File('data/chorales_nn.hdf5', "w", libver='latest') as f:
	for i in range(len(cX)):
		X3d = numpy.array(map(numpy.array, tuple(cX[i]))).transpose([1,0,2]) # seqlen * batchsize * featsize
		y3d = numpy.array(map(numpy.array, tuple(cy[i]))).transpose([1,0,2])
		assert X3d.shape[0] == SEQLEN
		assert X3d.shape[2] == len(cXall[0])
		f.create_dataset('chorale%dX' % (i + 1), X3d.shape, dtype="i", data=X3d)
		f.create_dataset('chorale%dy' % (i + 1), y3d.shape, dtype="i", data=y3d)
	f.create_dataset('metadata', (1,2), dtype="i", data=numpy.array([max(xpadding), max(ypadding)]))
print "written..."