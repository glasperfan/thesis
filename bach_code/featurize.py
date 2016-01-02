from helpers import * # includes music21
import math
import time
import sys
from ordered_set import OrderedSet
from random import shuffle
import numpy as npy
import h5py

#####
#
# wrangle1.py
#
# Goal: create a data representation of the entire Bach chorale set - that is, the 371 chorales 
# contained in the Riemenschieder edition.
# 
# Each chorale is quantiized to quarter notes, and then each beat reprsents a training example.
#
# INPUT Features:
# Local: [30 units]
# 	1) pitch -- represented as the distance from the tonic [22 units (range of soprano)]
# 	2) beat strength [1 unit (4, 2, 3, 1)]
# 	3) cadence? -- 1 for fermata, 0 otherwise [1 unit]
#	4) distance to next fermata (>= 0) [1 unit]
#	5) offset from beginning of the work [1 unit]
#	6) offset from the end of the work [1 unit]
#	7) time signature [1 - 4/4, 2 - 3/4, 3 - 12/8]
#	8) key signature [1 unit (number of flats/sharps, flats are negative)]
#	9) major/minor [1 unit (1 - major, 2 - minor)]
#
#
# OUTPUT Features:
#	A vector of length len(VOICINGS), where each entry I is a probability (0 < p < 1) that 
# 	the harmony should be VOICINGS[I].
#
# TOTAL: 16 input units, len(VOICINGS) output units
#
# The length of the VOICINGS dictionary mapping IDs to chords is determined by the function findUniqueATBs.
# This is run before extracting features to determine the VOICING mapping space.
#
# The goal is to provide accurate harmonization for each beat of the chorale.
#
#####


#########
# Globals
#########

# The 4 voices (and Part IDs) in a Bach chorale
PARTS = ['Soprano', 'Alto', 'Tenor', 'Bass']

# A dictionary
VOICINGS = {}

# The total range (measured in half-steps) of the chorales
TOTAL_RANGE = 95 # inclusive

# Since Torch is 1-based(?)
VECTOR_OFF = 0
VECTOR_ON = 1

# Range of each voice over all chorales 
RANGE = {
	'Soprano': {
		'max': 81,
		'min': 60
	},
	'Alto': {
		'max': 74,
		'min': 53,
	},
	'Tenor': {
		'max': 69,
		'min': 48
	},
	'Bass': {
		'max': 64,
		'min': 36
	}
}

# returns true if l <= v <= h
def in_range(v, l, h):
	return l <= v and v <= h

# Set of Chorales to work on
#WORK_SET_MIN = 0
WORK_SET_MAX = 30

# Preprocessed scores
QUANTIZED = []

# Use the @timing decorator to determine how long the function takes to run.
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f s' % (f.func_name, (time2-time1))
        return ret
    return wrap


#############################
# Featurizer helper functions
#############################

# Featurize the key signature (C major = 0, sharps are 1-7, flats are 8-14)
# This is done to ensure flats aren't given negative values
def feat_key(key_sig):
	return key_sig.sharps if key_sig.sharps >= 0 else 7 - key_sig.sharps

# Featurize the soprano note
def feat_pitch(n):
	return n.midi

# Featurize the beat strength value for the input note
def feat_beat(n):
	return n.beatStrength

# Featurize whether a cadence occurs on the note
def feat_cadence(n):
	return 1 if hasFermata(n) else 0

# Featurize the distance to the next fermata
# fermata_locations is 0 or 1
def feat_cadence_dist(n, index, fermata_locations):
	return 0 if hasFermata(n) else fermata_locations[index:].index(True)

# Featurize distance from the end of the chorale (approximately measured in measures)
def feat_offset_end(index, score_length):
	return int(math.floor((score_length - index) / 4.))

# Featurize the harmony as a tuple (ATB) at time i
def feat_chord(i, a, t, b):
	return a[i].midi, t[i].midi, b[i].midi

# Featurize harmony (s, a, t, and b are notes, while 'key_sig' is a key signature object)
def feat_harmony(s, a, t, b, key_obj):
	voicing = [s,a,t,b]
	rn = roman.romanNumeralFromChord(chord.Chord([s,a,t,b]), key_obj, True)
	rn_fig = simplify_harmony(rn)
	return rn_fig

# Converts all unknown harmonies to either a triad or a seventh chord
def simplify_harmony(rn):
	numeral = rn.romanNumeral
	if '7' in rn.figure:
		return numeral + '7'
	return numeral








class Featurizer(object):

	#
	#  1  pitch  22 23 beat str  26 27  cadence?  28 29  cadence_dist   59 60  off_end  160 161 time  162 163  key  174 175 mode 176
	# |------------|----------------|---------------|---------------------|----------------|-------------|-------------|-------------|
	#
	# We represent a single note as a list of features with their own ranges of indices.
	# [1, 24, 27, 30, 100, 161, 163, 176] --> lowest pitch, beat 2, no cadence, one beat from next cadence, 40 beats from end, 3/4, 6 flats, minor 
	#
	# NOTE: the indices above are simply examples. The correct indices are computed in analyze().
	# NOTE: indices are 1-based to comply with Torch

	# Initialize with the number of scores to analyze
	def __init__(self, num_scores=20):
		self.num_scores = num_scores
		self.indices = {}
		self.features = []
		self.harmonies = []
		self.max_index = 0
		self.original = [] 			# original, cleaned scores deposited here
		self.training_split = [] 	# training scores
		self.test_split = []		# test scores
		self.percentage_test = 0.1 	# percentage of scores to be in the test split

		self.data_dir = "raw_data/"
		self.output_dir = "data/"

		# Training examples created by featurize()
		self.X_train = []
		self.y_train = []
		self.X_test = []
		self.y_test = []


	# Collect all scores and preprocess them
	@timing
	def gather_scores(self):
		from os import listdir
		self.original = []
		for f in listdir(self.data_dir):
			if f.endswith(".xml"):
				self.original.append(converter.parse(self.data_dir + f))
		print "Gathered %d 4-part chorales." % len(self.original)
		
		return self.original

	# Analyze the chorales and determine the possible values for each feature
	@timing
	def analyze(self):
		self.analyzed = [] # to save time, we store the related objects to a score for featurizing

		# Reset feature sets
		self.keys = OrderedSet()
		self.key_modes = OrderedSet()
		self.times = OrderedSet()
		self.beats = OrderedSet()
		self.offset_ends = OrderedSet()
		self.cadence_dists = OrderedSet()
		self.cadences = OrderedSet(['cadence', 'no cadence'])
		self.harmonies = OrderedSet() # not a feature, this is the space of output classes
		# We don't need to check for any pitch feature space since we
		# already know the range of each voice.
		self.pitch = OrderedSet(range(RANGE['Soprano']['min'], RANGE['Soprano']['max'] + 1))
		self.pbefore = OrderedSet([0] + range(RANGE['Soprano']['min'], RANGE['Soprano']['max'] + 1)) # 0 is a dummy value
		self.pafter = OrderedSet([0] + range(RANGE['Soprano']['min'], RANGE['Soprano']['max'] + 1))
		self.harmonies = OrderedSet()
		# THIS ORDER MATTERS
		self.features = [('key', self.keys), ('mode', self.key_modes), ('time', self.times), \
						('beatstr', self.beats), ('offset', self.offset_ends), ('cadence_dists', self.cadence_dists), \
						('cadence?', self.cadences), ('pitch', self.pitch), ('pbefore', self.pitch), \
						('pafter', self.pitch)]

		for idx, score in enumerate(self.original):
			print idx
			# score-wide features
			S, A, T, B = getNotes(score.parts[0]), getNotes(score.parts[1]), getNotes(score.parts[2]), getNotes(score.parts[3])
			assert len(S) == len(A)
			assert len(A) == len(T)
			assert len(T) == len(B)
			time_sig, key_sig = getTimeSignature(score.parts[0]), getKeySignature(score.parts[0])
			key_obj = getKeyFromSignature(key_sig)
			fermata_locations = map(hasFermata, S)

			# Score-wide: Key (sharps, mode) and Time (num, denom)
			self.keys.add(feat_key(key_sig))
			self.key_modes.add(key_sig.mode)
			self.times.add((time_sig.numerator, time_sig.denominator))

			# Note-specific data
			for index, n in enumerate(S):
				# Beat strength
				self.beats.add(feat_beat(n))
				# Offset from the end
				self.offset_ends.add(feat_offset_end(index, len(S)))
				# Distance to next cadence
				self.cadence_dists.add(feat_cadence_dist(n, index, fermata_locations))
				# Harmony
				self.harmonies.add(feat_harmony(S[index], A[index], T[index], B[index], key_obj))

			# Store objects for featurizing
			self.analyzed.append((score, S, A, T, B, time_sig, key_sig, key_obj, fermata_locations))

		# Set feature indices
		i_max = 1
		for name, values in self.features:
			self.indices[name] = (i_max, i_max + len(values) - 1)
			i_max += len(values)
		self.max_index = i_max # record the highest index

	# Wrapper function for featurize_set():
	@timing
	def featurize(self):
		# Create train-test split
		training, test = self.training_test_split(self.analyzed)

		# Create training examples (note: score is a collection of objects)
		for idx, score in enumerate(training):
			sys.stdout.write("Featurizing #%d 	\r" % idx)
			sys.stdout.flush()
			X, y = self.featurize_score(score)
			self.X_train.append(X)
			self.y_train.append(y)
		print "Featurized training set."
		
		# Create test examples
		for idx, score in enumerate(test):
			sys.stdout.write("Featurizing #%d 	\r" % idx)
			sys.stdout.flush()
			X, y = self.featurize_score(score)
			self.X_test.append(X)
			self.y_test.append(y)
		print "Featurized training set."
		
		print "Training examples size: %d" % len(self.X_train)
		print "Test examples size: %d" % len(self.X_test)

		# Freeze for future use
		freezeObject(self.X_train, "X_train")
		freezeObject(self.y_train, "y_train")
		freezeObject(self.X_test, "X_test")
		freezeObject(self.y_test, "y_test")
		freezeObject(list(self.harmonies), "harmonies")
		freezeObject(self.indices, "indices")

		
	# After analysis, this generates the training examples (input vectors, output vectors)
	# As scores are examined, the indices of output chords are generated.
	def featurize_score(self, score_packet):
		# feature vectors
		X, y = [], []
		
		# unpack score objects
		score, S, A, T, B, time_sig, key_sig, key_obj, fermata_locations = score_packet

		# Create X vector and y output
		for index, n in enumerate(S):
			# Key
			f_key = self.keys.index(feat_key(key_sig)) + self.indices['key'][0]
			# Key mode
			f_mode = self.key_modes.index(key_sig.mode) + self.indices['mode'][0]
			# Time
			f_time = self.times.index((time_sig.numerator, time_sig.denominator)) + self.indices['time'][0]
			# Beat
			f_beat = self.beats.index(feat_beat(n)) + self.indices['beatstr'][0]
			# Offset end
			f_off_end = self.offset_ends.index(feat_offset_end(index, len(S))) + self.indices['offset'][0]
			# Cadence distance
			f_cadence_dist = self.cadence_dists.index(feat_cadence_dist(n, index, fermata_locations)) + self.indices['cadence_dists'][0]
			# Has cadence?
			f_cadence = feat_cadence(n) + self.indices['cadence?'][0]
			# Pitch
			f_pitch = self.pitch.index(feat_pitch(n)) + self.indices['pitch'][0]
			# Pitch before
			pitch_before = S[index - 1].midi if index > 0 else 0
			f_pbefore = self.pbefore.index(pitch_before) + self.indices['pbefore'][0]
			# Pitch after
			pitch_after = S[index + 1].midi if index + 1 < len(S) else 0
			f_pafter = self.pafter.index(pitch_after) + self.indices['pafter'][0]
 
			# Input vector (f_off_start deleted)
			input_vec = [f_key, f_mode, f_time, f_beat, f_off_end, f_cadence_dist, f_cadence, f_pitch, f_pbefore, f_pafter]

			# Output class, 1-indexed for Torch
			output_val = self.harmonies.index(feat_harmony(S[index], A[index], T[index], B[index], key_obj)) + 1

			X.append(input_vec)
			y.append(output_val)

		return X, y

	# Verify that the feature indices are all in the right ranges
	def verify(self):
		self.X_train, self.y_train = thawObject("X_train"), thawObject("y_train")
		self.X_test, self.y_test = thawObject("X_test"), thawObject("y_test")
		self.indices = thawObject('indices')
		self.harmonies = thawObject('harmonies')
		inputs = self.X_train + self.X_test
		outputs = self.y_train + self.y_test
		for i, score in enumerate(inputs):
			s_in = score
			s_out = outputs[i]
			for j, example in enumerate(s_in):
				output = s_out[j]

				# Note the order here corresponds with the order in which the example features were added
				features = ['key', 'mode', 'time', 'beatstr', 'offset', 'cadence_dists', 'cadence?', 'pitch', 'pbefore', 'pafter']
				for f_idx, feature in enumerate(features):
					try:
						assert in_range(example[f_idx], self.indices[feature][0], self.indices[feature][1])
					except:
						pass
				assert in_range(output, 1, len(self.harmonies))

	# Write 
	def write(self):
		for idx, score in enumerate(self.X_train):
			with h5py.File(self.output_dir + "train_%d.hdf5" % idx, "w", libver='latest') as f:
				X_matrix = npy.matrix(score)
				f.create_dataset("X", X_matrix.shape, dtype='i', data=X_matrix)
				y_vector = npy.array(self.y_train[idx])
				f.create_dataset("y", y_vector.shape, dtype='i', data=y_vector)
		
		for idx, score in enumerate(self.X_test):
			with h5py.File(self.output_dir + "test_%d.hdf5" % idx, "w", libver='latest') as f:
				X_matrix = npy.matrix(score)
				f.create_dataset("X", X_matrix.shape, dtype='i', data=X_matrix)
				y_vector = npy.array(self.y_test[idx])
				f.create_dataset("y", y_vector.shape, dtype='i', data=y_vector)

		# Freeze features for evaluation later on
		freezeObject(self.harmonies, "harmonies")

	# Split the quantized scores into a training and test split
	def training_test_split(self, score_list):
		shuffle(score_list)
		num_scores = len(score_list)
		split_point = int(num_scores * self.percentage_test)
		self.training_split = score_list[split_point:]
		self.test_split = score_list[:split_point]

		# Make sure there is no training, test_split overlap
		for s in self.training_split:
			for t in self.test_split:
				assert s != t

		return self.training_split, self.test_split

	def run(self):
		self.gather_scores()
		self.analyze()
		self.featurize()
		self.verify()
		self.write()

	def __str__(self):
		s = "\n---------- FEATURIZER RESULTS ----------\n"
		for name, values in self.features:
			s += name + ": " + str(values) + "\n"
		s += "\n"
		s += "Indices:\n"
		for name, values in self.features:
			s+= "'%s': %s\n" % (name, str(self.indices[name]))
		s += "\n"
		s += "Harmonies (%d total)\n%s\n" % (len(self.harmonies), self.harmonies)
		s += "Test-training split: %d training chorales, %d test chorales\n" % (len(self.training_split), len(self.test_split))
		s += "Test-training examples: %d for training, %d for test\n" % (len(self.X_train), len(self.X_test))
		s += "---------------------------------------\n"
		return s

	__repr__ = __str__

	

sf = Featurizer()
sf.run()
print sf


