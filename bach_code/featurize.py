from helpers import * # includes music21
import math
import time
import sys
import os
from ordered_set import OrderedSet
from glob import glob
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

# Featurize the key signature (sharps are positive, flats are negative values)
def feat_key(key_sig):
	return key_sig.sharps

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
	rn = roman.romanNumeralFromChord(chord.Chord([s,a,t,b]), key_obj)
	rn_fig = verify_harmony(rn, key_obj)
	return rn_fig

def get_extension(rn):
	return rn.figure[rn.figure.index(rn.romanNumeral) + len(rn.romanNumeral):]

# Human teaching of Roman numeral analysis
# this has poor practice written all over it, I know
basicRomanNumerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'i', 'ii', 'iio' 'iii', 'iv', 'v', 'vi', 'vii', 'viio']
basicExtensions = ['', '6', '64', '7', 'b7', '65', '43', '42']
def verify_harmony(rn, key_obj):
	if not os.path.isfile("_frozen/harmony_dict.txt"):
		freezeObject({}, "harmony_dict")
	harmony_dict = thawObject("harmony_dict")
	if rn.figure not in harmony_dict:
		for idx in [1,2,3]:
			if rn.figure[:idx] in basicRomanNumerals and rn.figure[idx:] in basicExtensions:
				harmony_dict[rn.figure] = (rn.figure[:idx], rn.figure[idx:])
				return harmony_dict[rn.figure]
		inversion_changes = {('b7', '5', '3') : 'b7',
							 ('b7', '4', '3') : 'b743',
							 ('b7', '3') : 'b7',
							 ('b7', '6', '5') : 'b765',
							 ('b7', '4', '2') : 'b742',
							 '765': '65',
							 '742' : '2',
							 '753' : '7',
							 '73': '7',
							 '653' : '65', 
							 '643': '43',
							 '642' : '42',
							 '63' : '63',
							 '64' : '64', 
							 '532': '2', 
							 '54' : '4',
							 '42' : '42'}
		for inv, correct_inv in inversion_changes.items():
			if _contains(rn.figure, inv):
				rml = rn.romanNumeral + 'o' if 'o' in rn.figure else rn.romanNumeral
				rml = 'b' + rml[1:] if '-' in rn.romanNumeral else rml
				rml = 'VI' if rml == 'bVI' and key_obj.mode == 'minor' else rml
				harmony_dict[rn.figure] = (rml, correct_inv)
				freezeObject(harmony_dict, "harmony_dict")
				print "Added %s as %s." % (rn.figure, harmony_dict[rn.figure])
				return harmony_dict[rn.figure]
		print "The current figure is %s." % rn.figureAndKey
		print "The pitches are %s %s %s %s." % rn.pitches
		new_numeral = raw_input("What should the new roman numeral be?: ")
		new_extension = raw_input("What should the new extension be?: ")
		harmony_dict[rn.figure] = (new_numeral, new_extension)
		freezeObject(harmony_dict, "harmony_dict")
	return harmony_dict[rn.figure]

def _contains(s, seq):
	for c in seq:
		if c not in s:
			return False
	return True



# Featurize melodic motion as an interval (an integer representing the half steps between two pitches)
# If n1 is a lower pitch than n2, the interval will be a positive value, and vice versa.
def feat_interval(n1, n2):
	return n2.midi - n1.midi








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
		for f in glob(self.data_dir + "*.xml"):
			self.original.append(converter.parse(f))
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
		self.intervals = OrderedSet()
		self.cadences = OrderedSet(['cadence', 'no cadence'])
		self.pitch = OrderedSet(range(RANGE['Soprano']['min'], RANGE['Soprano']['max'] + 1))
		self.numerals = OrderedSet() # output feature
		self.inversions = OrderedSet() # output feature
		# THIS ORDER MATTERS
		self.features = [('key', self.keys), ('mode', self.key_modes), ('time', self.times), \
						('beatstr', self.beats), ('offset', self.offset_ends), ('cadence_dists', self.cadence_dists), \
						('cadence?', self.cadences), ('pitch', self.pitch), ('ibefore', self.intervals), \
						('iafter', self.intervals), ('numeral_prev', self.numerals), ('inv_prev', self.inversions)]

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
				# Intervals
				if index > 0:
					self.intervals.add(feat_interval(S[index - 1], S[index]))
				# Harmony
				numeral, inversion = feat_harmony(S[index], A[index], T[index], B[index], key_obj)
				self.numerals.add(numeral)
				self.inversions.add(inversion)

			# Store objects for featurizing
			self.analyzed.append((score, S, A, T, B, time_sig, key_sig, key_obj, fermata_locations))

		# Add 'None' as an option for previous harmonies (i.e. to say there's no previous harmony for the first beat)
		self.numerals.add('None')
		self.inversions.add('None')
		# Add 'None' as an option for previous and future melodic intervals
		# (i.e. the first note has no previous note, so the 'interval before' is represented as 'None')
		self.intervals.add('None')

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
			sys.stdout.write("Featurizing #%d 	\r" % (idx + 1))
			sys.stdout.flush()
			X, y = self.featurize_score(score)
			self.X_train.append(X)
			self.y_train.append(y)
		print "Featurized training set."
		
		# Create test examples
		for idx, score in enumerate(test):
			sys.stdout.write("Featurizing #%d 	\r" % (idx + 1))
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
		freezeObject(list(self.numerals), "numerals")
		freezeObject(list(self.inversions), "inversions")
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
			# Melodic interval before
			ibefore = feat_interval(S[index - 1], S[index]) if index > 0 else 'None'
			f_ibefore = self.intervals.index(ibefore) + self.indices['ibefore'][0]
			# Melodic interval after
			iafter = feat_interval(S[index], S[index + 1]) if index < len(S) - 1 else 'None'
			f_iafter = f_pbefore = self.intervals.index(iafter) + self.indices['iafter'][0]
			# Previous harmony
			num_prev, inv_prev = feat_harmony(S[index - 1], A[index - 1], T[index - 1], B[index - 1], key_obj) if index > 0 else ('None', 'None')
			f_num_prev = self.numerals.index(num_prev) + self.indices['numeral_prev'][0]
			f_inv_prev = self.inversions.index(inv_prev) + self.indices['inv_prev'][0]
			# Input vector
			input_vec = [f_key, f_mode, f_time, f_beat, f_off_end, f_cadence_dist, f_cadence, f_pitch, \
						f_ibefore, f_iafter, f_num_prev, f_inv_prev]

			# Output class, 1-indexed for Torch
			f_num, f_prev = feat_harmony(S[index], A[index], T[index], B[index], key_obj)
			output_vec = [self.numerals.index(f_num) + 1, self.inversions.index(f_prev) + 1]

			X.append(input_vec)
			y.append(output_vec)

		return X, y

	# Verify that the feature indices are all in the right ranges
	def verify(self):
		print "Verifying indices..."
		self.X_train, self.y_train = thawObject("X_train"), thawObject("y_train")
		self.X_test, self.y_test = thawObject("X_test"), thawObject("y_test")
		self.indices = thawObject('indices')
		self.numerals = thawObject('numerals')
		self.inversions = thawObject('inversions')
		inputs = self.X_train + self.X_test
		outputs = self.y_train + self.y_test
		for i, score in enumerate(inputs):
			s_in = score
			s_out = outputs[i]
			for j, example in enumerate(s_in):
				numeral, inversion = s_out[j]

				# Note the order here corresponds with the order in which the example features were added
				features = ['key', 'mode', 'time', 'beatstr', 'offset', 'cadence_dists', 'cadence?', 'pitch', 'pbefore', 'pafter']
				for f_idx, feature in enumerate(features):
					try:
						assert in_range(example[f_idx], self.indices[feature][0], self.indices[feature][1])
					except:
						pass
				try:
					assert in_range(numeral, 1, len(self.numerals))
					assert in_range(inversion, 1, len(self.inversions))
				except:
					pass

	# Write 
	def write(self):
		print "Writing to %s..." % self.output_dir
		for idx, score in enumerate(self.X_train):
			with h5py.File(self.output_dir + "train_%d.hdf5" % idx, "w", libver='latest') as f:
				X_matrix = npy.matrix(score)
				f.create_dataset("X", X_matrix.shape, dtype='i', data=X_matrix)
				y_matrix = npy.matrix(self.y_train[idx])
				f.create_dataset("y", y_matrix.shape, dtype='i', data=y_matrix)
		
		for idx, score in enumerate(self.X_test):
			with h5py.File(self.output_dir + "test_%d.hdf5" % idx, "w", libver='latest') as f:
				X_matrix = npy.matrix(score)
				f.create_dataset("X", X_matrix.shape, dtype='i', data=X_matrix)
				y_matrix = npy.matrix(self.y_test[idx])
				f.create_dataset("y", y_matrix.shape, dtype='i', data=y_matrix)

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
		s += "Roman numerals (%d total)\n%s\n" % (len(self.numerals), self.numerals)
		s += "\n"
		s += "Inversions (%d total)\n%s\n" % (len(self.inversions), self.inversions)
		s += "\n"
		s += "Test-training split: %d training chorales, %d test chorales\n" % (len(self.training_split), len(self.test_split))
		s += "Test-training examples: %d for training, %d for test\n" % (len(self.X_train), len(self.X_test))
		s += "---------------------------------------\n"
		return s

	__repr__ = __str__

	

sf = Featurizer()
sf.run()
print sf
freezeObject(sf, 'featurizer')


