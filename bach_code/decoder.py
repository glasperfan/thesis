# decoder.py
# Generate the musical scores for a Bach chorale and its prediction harmonization

import numpy as np
from music21 import *
from logit import load_dataset, encode, score_with_padding
from ordered_set import OrderedSet
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from helpers import *
from music21 import * # for score creation

# Load data
trainXc, trainyc = load_dataset("train", "data/chorales_rnn.hdf5")
devXc, devyc = load_dataset("dev", "data/chorales_rnn.hdf5")
testXc, testyc = load_dataset("test", "data/chorales_rnn.hdf5")
# Remove Oracle features
trainXc = [X[:, range(0,10)] for X in trainXc]
devXc = [X[:, range(0,10)] for X in devXc]
testXc = [X[:, range(0,10)] for X in testXc]
stack = lambda x1, x2: np.vstack((x1, x2))
hstack = lambda x1, x2: np.hstack((x1, x2))
Xtrain = stack(reduce(stack, trainXc), reduce(stack, devXc))
ytrain = hstack(reduce(hstack, trainyc), reduce(hstack, devyc))
Xtest, ytest = reduce(stack, testXc), reduce(hstack, testyc)


def removePadding(X, y, ypadding):
	outX, outy = [], []
	for idx, row in enumerate(X):
		if y[idx] != ypadding:
			outX.append(X[idx])
			outy.append(y[idx])
	return np.matrix(outX), np.array(outy)

yp = ytest.max() # ypadding
Xtrain, ytrain = removePadding(Xtrain, ytrain, yp)
Xtest, ytest = removePadding(Xtest, ytest, yp)
Xall = np.vstack((Xtrain, Xtest))

# Encode data
encoder = OneHotEncoder()
encoder.fit(Xall)
Xtrainsparse = encoder.transform(Xtrain)
Xtestsparse = encoder.transform(Xtest)

# Fit data
print "fitting data..."
RF = RandomForestClassifier(10, "entropy", None)
RF.fit(Xtrainsparse, ytrain)
preds = range(len(testXc))
scores = np.zeros(len(testXc))
for i in range(len(testXc)):
	X, y = removePadding(testXc[i], testyc[i], yp)
	Xsparse = encoder.transform(X)
	preds[i] = RF.predict(Xsparse)
	scores[i] = RF.score(Xsparse, y)
	print "Chorale #%d: %.2f%%" % (i, scores[i] * 100.0)
print "AVERAGE SCORE: %.2f%%" % (scores.mean() * 100.0)




# Materials
harm_dict = thawObject('harmdict') # { (7, 18, 1, 1, 12): 1980}
harm_dict_rev = {v: k for k, v in harm_dict.items()} # { 1980: (7, 18, 1, 1, 12)}
roots = list(thawObject('roots'))
bases = list(thawObject('bases'))
inversions = list(thawObject('inversions'))
altos = list(thawObject('alto_range'))
tenors = list(thawObject('tenor_range'))

# full harmonization index --> subtask index tuple
lookup = lambda x: harm_dict_rev[x]
# subtask index tuple --> subtask feature tuple (what the indices represent)
lookup2 = lambda x: (roots[x[0] - 1], bases[x[1] - 1], inversions[x[2] - 1], altos[x[3] - 1], tenors[x[4] - 1])

# Perform reverse lookup on the chorale to convert extract the predictions for each subtask at each time step
def perform_lookup(chorale_idx):
	original_score = converter.parse('data/test_scores/%d.xml' % chorale_idx)
	y_predicted = preds[chorale_idx]
	y_predicted = map(lookup, y_predicted)
	y_predicted = map(lookup2, y_predicted)
	return original_score, y_predicted

#### MAKE THE SCORE ####
def make_score(original_score, y_predicted, chorale_idx):
	s = stream.Score()
	soprano = original_score.parts[0]
	time_sig, key_sig = getTimeSignature(soprano), getKeySignature(soprano)
	key = getKeyFromSignature(key_sig)
	tonic = key.tonic.midi # Do all arithmetic in MIDI
	bpm = time_sig.numerator # beats per measure
	pickup_len = len(getMeasures(soprano)[0].notes)
	pickup_rest_len = bpm - pickup_len

	# Create parts
	def addPart(p_id, instrument, clef):
		new_part = stream.Part()
		new_part.offset = 0.0
		new_part.id = p_id
		new_part.append(instrument)
		# Clef, key, time
		new_part.append(clef)
		new_part.append(key_sig)
		new_part.append(time_sig)
		if pickup_rest_len > 0: # for pickup measure, if exists
			new_part.append(note.Rest(quarterLength=pickup_rest_len))
		s.insert(0, new_part)
		return new_part

	S = addPart(PARTS[0], instrument.Soprano(), clef.TrebleClef()) # Soprano
	A = addPart(PARTS[1], instrument.Alto(), clef.TrebleClef()) # Alto
	T = addPart(PARTS[2], instrument.Tenor(), clef.BassClef()) # Tenor
	B = addPart(PARTS[3], instrument.Bass(), clef.BassClef()) # Bass

	# Add the original soprano
	for nt in soprano.flat.notes:
		S.append(nt)

	# Add the other voices
	# Prediction for time t: (0, (0, 4, 7), 0, 0, 7)
	o8vb = lambda x, y: x - y * 12 # down an octave
	o8va = lambda x, y: x + y * 12 # up an octave

	for ts_pred in y_predicted: # iterate over the predictions for each time step
		alto_p = note.Note(o8vb(tonic + ts_pred[3], 1), quarterLength=1) 
		A.append(alto_p)
		tenor_p = note.Note(o8vb(tonic + ts_pred[4], 1), quarterLength=1) 
		T.append(tenor_p)
		bass_midi = tonic + ts_pred[0] + ts_pred[1][ts_pred[2]] # tonic + root_distance from tonic + inversion
		bass_p = note.Note(o8vb(bass_midi, 2), quarterLength=1)
		B.append(bass_p)

	# Create measures
	S.makeMeasures(inPlace=True)
	A.makeMeasures(inPlace=True)
	T.makeMeasures(inPlace=True)
	B.makeMeasures(inPlace=True)

	# Delete any pickups
	if pickup_rest_len > 0:
		S[0].remove(S[0].notesAndRests[0])
		A[0].remove(A[0].notesAndRests[0])
		T[0].remove(T[0].notesAndRests[0])
		B[0].remove(B[0].notesAndRests[0])

	# Write the score
	output_dir = '/Users/hzabriskie/Documents/Thesis/thesis/bach_code/examples'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	s.write('musicxml', '%s/%d.xml' % (output_dir, chorale_idx))


## RUN ##
for i in range(len(testXc)):
	sc, y_pred = perform_lookup(i)
	make_score(sc, y_pred, i)












