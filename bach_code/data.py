from helpers import * # includes music21
import gct
import math
import time
import sys
from ordered_set import OrderedSet
from glob import glob
from random import shuffle
import numpy as npy
import h5py
import os

#####
#
# data.py
#
# Goal: create a data representation of the entire Bach chorale set - that is, the 371 chorales 
# contained in the Riemenschieder edition.
# 
# Each chorale is quantiized to quarter notes, and then each beat reprsents a training example.
#
# Input data contains melody note a time t. Features include score-wide properties like key and time,
# as well as beat-specific information such as beat strength, offset from the end of the chorale, distance to
# the next cadence, whether a cadence occurs on that beat, and the melodic interval before and after the 
# current melody pitch.
#
# There are output datasets from the following properties:
# - Root note, relative to the tonic
# - Inversions (root, 1st, 2nd, or 3rd inversion)
# - Base, or chord function (major, minor, etc.), defined as a set of intervals away from the tonic
#
#
#
#
#
#####


##########################
# General helper functions
##########################

# Split a list randomly into two lists with a proportion of [pg : 1 - pg]
def split(lst, pg):
		shuffle(lst)
		split_point = int(len(lst) * pg)
		return lst[:split_point], lst[split_point:]

def _contains(s, seq):
	for c in seq:
		if c not in s:
			return False
	return True

# Use the @timing decorator to determine how long the function takes to run.
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f s' % (f.func_name, (time2-time1))
        return ret
    return wrap



class Featurizer(object):
	#
	# Converts chorales into a matrix of feature indices. Each vector in a matrix represents a specific beat within
	# a chorale. Note that indices are 1-based to comply with Torch. 
	#

	# Initialize with the number of scores to analyze
	def __init__(self):
		self.percentage_train = 0.8 # percentage of scores to be in the test split
		self.percentage_dev = 0.5 	# percentage of the test set to be used a dev set
		self.data_dir = "raw_data/"
		self.output_dir = "data/"

		# Features
		self.keys = OrderedSet()
		self.modes = OrderedSet()
		self.times = OrderedSet()
		self.beats = OrderedSet()
		self.offsets = OrderedSet()
		self.cadence_dists = OrderedSet()
		self.cadences = OrderedSet()
		self.pitches = OrderedSet()
		self.intervals = OrderedSet()
		self.roots = OrderedSet()
		self.inversions = OrderedSet()
		self.bases = OrderedSet()
		self.altos = OrderedSet()
		self.tenors = OrderedSet()

		# THIS ORDER MATTERS
		self.input_features = [self.keys, self.modes, self.times, self.beats, self.offsets, self.cadence_dists, \
								self.cadences, self.pitches, self.intervals, self.intervals, self.roots, \
								self.bases, self.inversions]
		self.output_features = [self.roots, self.bases, self.inversions, self.altos, self.tenors]

	# Collect all preprocessed scores
	@timing
	def gather_scores(self):
		from os import listdir
		self.original = []
		for f in glob(self.data_dir + "*.xml"):
			score = converter.parse(f)
			if score.parts[0].quarterLength > 300: # removes on excessively long score
				continue
			self.original.append(score)
		print "Gathered %d 4-part chorales." % len(self.original)
		
		return self.original

	# Create X and y matrices of features for each chorale
	@timing
	def analyze(self):
		print "Analyzing..."
		self.analyzed = [] # to save time, we store the related objects to a score for featurizing
		Xvalues, yvalues = [], []

		# Create X and y matrices for each chorale
		for idx, score in enumerate(self.original):
			sys.stdout.write("Analyzing #%d 	\r" % (idx + 1))
			sys.stdout.flush()
			# score-wide features
			S, A, T, B = getNotes(score.parts[0]), getNotes(score.parts[1]), getNotes(score.parts[2]), getNotes(score.parts[3])
			assert len(S) == len(A)
			assert len(A) == len(T)
			assert len(T) == len(B)
			time_sig, key_sig = getTimeSignature(score.parts[0]), getKeySignature(score.parts[0])
			key_obj = getKeyFromSignature(key_sig)
			tonic = key_obj.tonic.midi
			fermata_locations = map(hasFermata, S)

			# Input/target data for each chorale
			Xc, yc = [], []

			# Create X vector and y output
			for index, n in enumerate(S):
				# [0]: Key
				v_key = key_sig.sharps
				self.keys.add(v_key)
				# [1]: Mode
				v_mode = key_sig.mode
				self.modes.add(v_mode)
				# [2]: Time
				v_time = (time_sig.numerator, time_sig.denominator)
				self.times.add(v_time)
				# [3]: Beat strength
				v_beat = n.beatStrength
				self.beats.add(n.beatStrength)
				# [4]: Offset end
				v_off_end = int(math.floor((len(S) - index) / 4.))
				self.offsets.add(v_off_end)
				# [5]: Cadence distance
				v_cadence_dist = 0 if hasFermata(n) else fermata_locations[index:].index(True)
				self.cadence_dists.add(v_cadence_dist)
				# [6]: Is a point of cadence
				v_cadence = 1 if hasFermata(n) else 0
				self.cadences.add(v_cadence)
				# [7]: Soprano pitch (relative to key signature)
				v_pitch = (n.midi - tonic) % 12
				self.pitches.add(v_pitch)
				# [8]: Interval to previous melody note
				v_ibefore = S[index].midi - S[index - 1].midi if index > 0 else 'None'
				self.intervals.add(v_ibefore)
				# [9]: Interval to next melody note
				v_iafter = S[index + 1].midi - S[index].midi if index < len(S) - 1 else 'None'
				self.intervals.add(v_iafter)
				# [10]: root at time t-1
				# [11]: base at time t-1
				# [12]: inversion at time t-1
				timetminus1 = yc[-1] if len(yc) > 0 else ('*padding*', '*padding*', '*padding*')
				v_root_prev = timetminus1[0] # NOTE THE ORDER
				v_base_prev = timetminus1[1]
				v_inv_prev = timetminus1[2]
				
				# Output vector
				# [0]: root
				# [1]: base
				# [2]: inversion
				consonance = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0] # see gct module
				v_root, v_inv, v_base = gct.GCT(tonic, consonance, chord.Chord([B[index], T[index], A[index], S[index]]))
				self.roots.add(v_root)
				self.bases.add(v_base)
				self.inversions.add(v_inv)
				# [3]: Alto pitch (relative to key signature)
				v_alto = (A[index].midi - tonic) % 12
				self.altos.add(v_alto)
				# [4]: Tenor pitch (relative to key signature)
				v_tenor = (T[index].midi - tonic) % 12
				self.tenors.add(v_tenor)
 
				# Input vector
				input_vec = [v_key, v_mode, v_time, v_beat, v_off_end, v_cadence_dist, v_cadence, \
							 v_pitch, v_ibefore, v_iafter, v_root_prev, v_base_prev, v_inv_prev]
				output_vec = [v_root, v_base, v_inv, v_alto, v_tenor]

				Xc.append(input_vec)
				yc.append(output_vec)

			self.analyzed.append((Xc, yc, score, idx))
			Xvalues.append(Xc)
			yvalues.append(yc)

		# Add the 'n/a' option
		for feature_space in self.input_features + self.output_features:
			feature_space.add('*padding*')

		freezeObject(Xvalues, 'Xvalues')
		freezeObject(yvalues, 'yvalues')
		freezeObject(self.roots, 'roots')
		freezeObject(self.bases, 'bases')
		freezeObject(self.inversions, 'inversions')
		freezeObject(self.altos, "alto_range")
		freezeObject(self.tenors, "tenor_range")
		freezeObject(self.input_features, "input_features")
		freezeObject(self.output_features, "output_features")
	
	# After calling self.analyze, this converts the X and y matrices to vectors of feature indices
	# As scores are examined, the indices of output chords are generated.
	@timing
	def featurize(self):
		print "Featurizing..."
		self.featurized = []

		# Set the indices
		self.input_indices = []
		max_index = 1
		for feature_space in self.input_features:
			self.input_indices.append((max_index, max_index + len(feature_space) - 1))
			max_index += len(feature_space)

		for Xc, yc, score, idx in self.analyzed:
			Xcf, ycf = [], []
			for vec in Xc:
				fvec = []
				for fidx, feature_space in enumerate(self.input_features):
					f_feature = feature_space.index(vec[fidx])
					fvec.append(f_feature + self.input_indices[fidx][0])
				Xcf.append(fvec)
			for vec in yc:
				fvec = []
				for fidx, feature_space in enumerate(self.output_features):
					fvec.append(feature_space.index(vec[fidx]) + 1)
				ycf.append(fvec)
			self.featurized.append((npy.matrix(Xcf), npy.matrix(ycf), score, idx))


	# Verify that the feature indices are all in the right ranges
	@timing
	def verify(self):
		print "Verifying indices..."
		for Xcf, ycf, score, idx in self.featurized:
			for fidx in range(Xcf.shape[1]):
				assert Xcf[:, fidx].min() >= self.input_indices[fidx][0]
				assert Xcf[:, fidx].max() <= self.input_indices[fidx][1]
				if fidx > 0:
					assert Xcf[:, fidx - 1].max() < Xcf[:, fidx].min() 
			for fidx in range(ycf.shape[1]):
				assert ycf[:, fidx].min() >= 1
				assert ycf[:, fidx].max() <= len(self.output_features[fidx])

	# Split the chorales into training, dev, and test groups
	@timing
	def train_test_split(self):
		self.train, remaining = split(self.featurized, self.percentage_train)
		self.dev, self.test = split(remaining, self.percentage_dev)
		print "Training, dev, and tests sets contain %d, %d, %d chorales, respectively." \
				% (len(self.train), len(self.dev), len(self.test))

	# Create the aggregate datasets
	@timing
	def aggregrate(self):
		stack = lambda x1, x2: npy.vstack((x1, x2))
		self.trainX, self.trainy = [x for (x, y, sc, idx) in self.train], [y for (x, y, sc, idx) in self.train]
		self.devX, self.devy = [x for (x, y, sc, idx) in self.dev], [y for (x, y, sc, idx) in self.dev]
		self.testX, self.testy = [x for (x, y, sc, idx) in self.test], [y for (x, y, sc, idx) in self.test]
		self.trainXall, self.trainyall = reduce(stack, self.trainX), reduce(stack, self.trainy)
		self.devXall, self.devyall = reduce(stack, self.devX), reduce(stack, self.devy)
		self.testXall, self.testyall = reduce(stack, self.testX), reduce(stack, self.testy)
		self.dataXall = stack(stack(self.trainXall, self.devXall), self.testXall)
		self.datayall = stack(stack(self.trainyall, self.devyall), self.testyall)
	# Write 
	@timing
	def write(self):
		print "Writing to %s folder." % self.output_dir
		with h5py.File(self.output_dir + "chorales.hdf5", "w", libver="latest") as f:
			# Write accumulated chorales
			f.create_dataset("Xtrain", self.trainXall.shape, dtype="i", data=self.trainXall)
			f.create_dataset("ytrain", self.trainyall.shape, dtype="i", data=self.trainyall)
			f.create_dataset("Xdev", self.devXall.shape, dtype="i", data=self.devXall)
			f.create_dataset("ydev", self.devyall.shape, dtype="i", data=self.devyall)
			f.create_dataset("Xtest", self.testXall.shape, dtype="i", data=self.testXall)
			f.create_dataset("ytest", self.testyall.shape, dtype="i", data=self.testyall)
			# Write every chorale into train/dev/test sets
			with open('data/chorale_index.txt', 'w') as m:
				m.write("TRAINING SET\n")
				for idx, (X, y) in enumerate(zip(self.trainX, self.trainy)):
					f.create_dataset("train/chorale%d_X" % idx, X.shape, dtype="i", data=X)
					f.create_dataset("train/chorale%d_y" % idx, y.shape, dtype="i", data=y)
					m.write("%d\t %s\n" % (idx, self.train[idx][2].metadata.title))
				m.write("VALIDATION SET\n")
				for idx, (X, y) in enumerate(zip(self.devX, self.devy)):
					f.create_dataset("dev/chorale%d_X" % idx, X.shape, dtype="i", data=X)
					f.create_dataset("dev/chorale%d_y" % idx, y.shape, dtype="i", data=y)
					m.write("%d\t %s\n" % (idx, self.dev[idx][2].metadata.title))
				m.write("TEST SET\n")
				for idx, (X, y) in enumerate(zip(self.testX, self.testy)):
					f.create_dataset("test/chorale%d_X" % idx, X.shape, dtype="i", data=X)
					f.create_dataset("test/chorale%d_y" % idx, y.shape, dtype="i", data=y)
					m.write("%d\t %s\n" % (idx, self.test[idx][2].metadata.title))
			# Write every chorale individually
			for Xcf, ycf, score, idx in self.featurized:
				f.create_dataset("chorale%d_X" % idx, Xcf.shape, dtype="i", data=Xcf)
				f.create_dataset("chorale%d_y" % idx, ycf.shape, dtype="i", data=ycf)

		# Save test scores for future use
		test_scores = [sc for (x, y, sc, idx) in self.test]
		test_dir = '/Users/hzabriskie/Documents/Thesis/thesis/bach_code/data/test_scores'
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
		for idx, sc in enumerate(test_scores):
			sc.write('musicxml', test_dir + '/' + str(idx) + '.xml')


	def run(self):
		self.gather_scores()
		self.analyze()
		self.featurize()
		self.verify()
		self.train_test_split()
		self.aggregrate()
		self.write()

	

sf = Featurizer()
sf.run()
