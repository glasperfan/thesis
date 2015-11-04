from music21 import *
import math
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


#########
# HELPERS
#########

def getTimeSignature(_stream):
	return _stream.flat.getElementsByClass('TimeSignature')[0]

def getKeySignature(_stream):
	return _stream.flat.getElementsByClass('KeySignature')[0]

def getMeasures(_stream):
	return _stream.getElementsByClass(stream.Measure)

def getNotes(_stream):
	return _stream.flat.notes

def getPart(_score, _partID):
	return _score.parts[PARTS.index(_partID)]

def isMeasure(ms):
	return isinstance(ms, stream.Measure)

def isNote(n):
	return isinstance(n, note.Note)

def isChord(c):
	return isinstance(c, chord.Chord)

def length(n):
	return n.duration.quarterLength

def hasFermata(n):
	return any(type(x) is expressions.Fermata for x in n.expressions)




###############
# Preprocessing
###############

def removeTies(_score, _partID):
	part = getPart(_score, _partID)
	ts = getTimeSignature(part)

	for ms in getMeasures(part):
		for n in ms:
			if isNote(n):
				if n.tie is not None:
					n.tie = None

# Remove rests from a stream by lengthening the previous note
def removeRests(_stream):
	nar = _stream.notesAndRests
	for nr in nar:
		if isinstance(nr, note.Rest):
			prev_note = nar[nar.index(nr) - 1]
			prev_note.quarterLength += nr.quarterLength
			_stream.remove(nr)
	return _stream

# Transforms the specified part in the score into uniform quarter notes
def quantize(_score, _partID):
	part = getPart(_score, _partID)
	ts = getTimeSignature(part)
	measures = getMeasures(part)

	# TODO: catch the 12/8 case
	if ts.ratioString is '12/8':
		raise Exception("12/8 chorale")

	# first, remove any ties
	removeTies(_score, _partID)

	# iterate over each measure
	for ms in measures:
		index = 0
		notes = getNotes(removeRests(ms))
		note_num = len(notes)
		# iterate over each note
		while index < note_num:
			n = notes[index]
			if isChord(n):
				top_note = note.Note(n[-1].pitch.nameWithOctave, quarterLength=n[-1].quarterLength)
				ms.insert(n.offset, top_note)
				ms.remove(n)

			# skip quarter notes
			if length(n) == 1:
				index += 1

			# combine notes that add up to one beat
			# 1/8G + 1/16A + 1/16B --> 1/4G
			elif length(n) < 1:# and index < len(notes) - 1:
				context = [n]
				context_duration = length(n)
				while context_duration % 1 != 0:
					next_note = notes[index + len(context)]
					context.append(next_note)
					context_duration += length(next_note)
				if context_duration == 1.0:
					n.duration.quarterLength = 1.0
					for other_note in context[1:]:
						ms.remove(other_note)
				# keep the 1st and last note
				elif context_duration == 2.0:
					n.duration.quarterLength = 1.0
					context[-1].duration.quarterLength = 1.0
					map(lambda x : ms.remove(x), context[1:-1])
				else:
					print "Weird syncopation"

			# break down half notes, dotted halves, whole notes (keeping fermatas)
			elif length(n) > 1 and length(n) % 1 == 0:
				total_beats = int(length(n))
				n.quarterLength = 1.0
				for beat in range(1, total_beats):
					new_note = note.Note(n.pitch.nameWithOctave, quarterLength=1.0)
					new_note.expressions = n.expressions
					# avoid unneccessary accidentals
					if n.accidental != None:
						new_note.accidental = None
					ms.insert(n.offset + beat, new_note)

			# dotted quarter and eighth (or 2 sixteenths) case
			elif length(n) > 1:
				context = [n]
				context_duration = length(n)
				while context_duration % 1 != 0:
					next_note = notes[index + len(context)]
					context.append(next_note)
					context_duration += length(next_note)
				n.duration.quarterLength = 1.0
				context[1].duration.quarterLength = 1.0 # the second note
				context[1].offset = n.offset + 1.0
				for other_note in context[2:]:
					ms.remove(other_note)

			else:
				print "Error occurred in quanitization."
				print ms
				print n
				_score.show()
				raise Exception(part, ms, n, n.offset)

			# Reset these values since notes may have been added/deleted
			notes = getNotes(ms)
			note_num = len(notes)


## Helper function: find the global minimum and maximum pitch for each voice over the entire set of chorales
# <scores>: a list of score objects
# <partID>: a partID (see PARTS)
def choraleRange(scores, partID):
	p = analysis.discrete.Ambitus()
	globalMin = None
	globalMax = None
	for score in scores:
		part = getPart(score, partID)
		pitchMin, pitchMax = p.getPitchSpan(part)
		if scores.index(score) == 0:
			globalMin = pitchMin
			globalMax = pitchMax
		if pitchMin.midi < globalMin.midi:
			globalMin = pitchMin
			print "new min of %d found at index %d" % (globalMin.midi, scores.index(score))
		if pitchMax.midi > globalMax.midi:
			globalMax = pitchMax
			print "new max of %d found at index %d" % (globalMax.midi, scores.index(score))
	return globalMin, globalMax

def allChoralesRange(chorales):
	global PARTS
	ranges = {}
	for part in PARTS:
		ranges[part] = dict()
		print "analyzing %s" % part
		gmin, gmax = choraleRange(chorales, part)
		ranges[part]['min'] = gmin
		ranges[part]['min_midi'] = gmin.midi
		ranges[part]['max'] = gmax
		ranges[part]['max_midi'] = gmax.midi
	print pp(ranges)

### Wrapper function: calls choraleRange on all chorales for all voices
### Correlated results with http://www.ofai.at/~soren.madsen/daimi/harmreport.pdf

# Soprano: 		[60; 81]	range: 22 		--all ranges inclusive
# Alto: 		[53; 74]	range: 22
# Tenor: 		[48; 69]	range: 22
# Bass: 		[36; 64]	range: 29

# Total range: 95 (inclusive)

# My results:
# {'Alto': {'max': <music21.pitch.Pitch D5>,
#           'max_midi': 74,
#           'min': <music21.pitch.Pitch F3>,
#           'min_midi': 53},
#  'Bass': {'max': <music21.pitch.Pitch E4>,
#           'max_midi': 64,
#           'min': <music21.pitch.Pitch C2>,
#           'min_midi': 36},
#  'Soprano': {'max': <music21.pitch.Pitch A5>,
#              'max_midi': 81,
#              'min': <music21.pitch.Pitch C4>,
#              'min_midi': 60},
#  'Tenor': {'max': <music21.pitch.Pitch A4>,
#            'max_midi': 69,
#            'min': <music21.pitch.Pitch C3>,
#            'min_midi': 48}}




class SopranoFeaturizer(object):

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
	def __init__(self, num_scores=371):
		self.num_scores = num_scores
		self.keys = OrderedSet() # sets must be ordered to ensure accurate indexing
		self.key_modes = OrderedSet()
		self.time_sigs = OrderedSet()
		self.beats = OrderedSet()
		self.offset_ends = OrderedSet()
		self.pitches = OrderedSet()
		self.chords = OrderedSet()
		self.cadence_dists = OrderedSet()
		self.cadences = OrderedSet(['cadence', 'no cadence'])
		self.indices = {}
		self.max_index = 0
		self.ordering = {}
		self.quantized = [] 		# quantized scores deposited here
		self.training_split = [] 	# training scores
		self.test_split = []		# test scores
		self.percentage_test = 0.1 	# percentage of scores to be in the test split
		self.analyzed = False		# stage 1
		self.featurized = False		# stage 2
		self.verified = False		# stage 3

		# Training examples created by featurize()
		self.X_train = []
		self.y_train = []
		self.X_test = []
		self.y_test = []

		# NOTE: this should allow for flexibility in the ordering of feature indices
		self.ordering['pitch'] = self.pitches
		self.ordering['beat_str'] = self.beats
		self.ordering['cadence?'] = self.cadences
		self.ordering['cadence_dist'] = self.cadence_dists
		self.ordering['offset_end'] = self.offset_ends
		self.ordering['time'] = self.time_sigs
		self.ordering['key'] = self.keys
		self.ordering['mode'] = self.key_modes

	# Collect all scores and preprocess them
	def gather_scores(self):
		iterator = corpus.chorales.Iterator(1, self.num_scores, numberingSystem = 'riemenschneider')
		self.quantized = []
		for score in iterator:
			if len(score.parts) == 4:
				# quantize
				self.quantize_score(score)
				self.quantized.append(score)

	# Analyze the chorales and determine the possible values for each feature
	def analyze(self):
		self.gather_scores()
		for score in self.quantized:
			if len(score.parts) != 4:
				continue		

			# score-wide features
			soprano = score.parts[0]
			alto = getNotes(score.parts[1])
			tenor = getNotes(score.parts[2])
			bass = getNotes(score.parts[3])
			time_signature = getTimeSignature(soprano)
			key_signature = getKeySignature(soprano)
			notes_lst = getNotes(soprano)
			last_note = notes_lst[-1]
			fermata_locations = map(hasFermata, notes_lst)

			# Key
			self.keys.add(key_signature.sharps)
			self.key_modes.add(key_signature.mode)

			# Time - as a tuple (num, denom)
			self.time_sigs.add((time_signature.numerator, time_signature.denominator))

			# Note-specific data
			for index, n in enumerate(notes_lst):

				# Beat strength
				self.beats.add(self.get_beat_str(n))

				# Offset from the end
				self.offset_ends.add(self.get_offset_end(n, last_note))

				# Distance to next cadence
				self.cadence_dists.add(self.get_cadence_dist(n, index, notes_lst, fermata_locations))

				# Pitch
				self.pitches.add(self.get_pitch(n))

				# Harmony
				self.chords.add(self.get_chord(index, alto, tenor, bass))

			# Set feature indices
			i_max = 1
			for feature in self.ordering.keys():
				self.indices[feature] = (i_max, i_max + len(self.ordering[feature]))
				i_max += len(self.ordering[feature]) + 1
			self.max_index = i_max # record the highest index

		# Now we can featurize
		self.analyzed = True

	# Wrapper function for featurize_set():
	def featurize(self):
		if not self.analyzed:
			raise Exception("Must call analyze first.")

		# Create train-test split
		training, test = self.training_test_split(self.quantized)

		# Create training examples
		self.X_train, self.y_train = self.featurize_set(training)
		
		# Create test examples
		self.X_test, self.y_test = self.featurize_set(test)
		
		print "Training examples size: %d" % len(self.X_train)
		print "Test examples size: %d" % len(self.X_test)

		
	# After analysis, this generates the training examples (input vectors, output vectors)
	# As scores are examined, the indices of output chords are generated.
	def featurize_set(self, scores):
		if not self.analyzed:
			raise Exception("Must call analyze first.")
		
		X, y = [], []
		for score in scores:
			# voices
			soprano = score.parts[0]
			alto = getNotes(score.parts[1])
			tenor = getNotes(score.parts[2])
			bass = getNotes(score.parts[3])

			# score-wide features
			soprano_notes = getNotes(soprano)
			time_sig = getTimeSignature(soprano)
			key_sig = getKeySignature(soprano)
			last_note = soprano_notes[-1]
			fermata_locations = map(hasFermata, soprano_notes)

			# Note-specific data
			for index, n in enumerate(soprano_notes):

				# input vector
				# NOTE: changing the order here will affect verify
				input_vec = []
				input_vec.append(self.pitches.index(self.get_pitch(n)) + self.indices['pitch'][0])
				input_vec.append(self.beats.index(self.get_beat_str(n)) + self.indices['beat_str'][0])
				input_vec.append(self.get_iscadence(n) + self.indices['cadence?'][0])
				input_vec.append(self.cadence_dists.index(self.get_cadence_dist(n, index, soprano_notes, fermata_locations)) + self.indices['cadence_dist'][0])
				input_vec.append(self.offset_ends.index(self.get_offset_end(n, last_note)) + self.indices['offset_end'][0])
				input_vec.append(self.time_sigs.index( (time_sig.numerator, time_sig.denominator) ) + self.indices['time'][0])
				input_vec.append(self.keys.index(key_sig.sharps) + self.indices['key'][0])
				input_vec.append(self.key_modes.index(key_sig.mode) + self.indices['mode'][0])

				# +1 since Torch must be 1-indexed
				output_val = self.chords.index(self.get_chord(index, alto, tenor, bass)) + 1

				X.append(input_vec)
				y.append(output_val)

		self.featurized = True
		return X, y

	# Verify that the feature indices are all in the right ranges
	def verify(self):
		if not self.analyzed or not self.featurized:
			raise Exception("Analyze and featurize first.")

		inputs = self.X_train + self.X_test
		outputs = self.y_train + self.y_test
		for index, example in enumerate(inputs):
			output = outputs[index]

			# Note the order here corresponds with the order in which the example features were added
			assert in_range(example[0], self.indices['pitch'][0], self.indices['pitch'][1])
			assert in_range(example[1], self.indices['beat_str'][0], self.indices['beat_str'][1])
			assert in_range(example[2], self.indices['cadence?'][0], self.indices['cadence?'][1])
			assert in_range(example[3], self.indices['cadence_dist'][0], self.indices['cadence_dist'][1])
			assert in_range(example[4], self.indices['offset_end'][0], self.indices['offset_end'][1])
			assert in_range(example[5], self.indices['time'][0], self.indices['time'][1])
			assert in_range(example[6], self.indices['key'][0], self.indices['key'][1])
			assert in_range(example[7], self.indices['mode'][0], self.indices['mode'][1])
			assert in_range(output, 0, len(self.chords))

		self.verified = True

	def write(self):
		if not self.analyzed or not self.featurized or not self.verified:
			raise Exception("Analyze, featurize, and verify first.")

		X_train_npy = npy.matrix(self.X_train)
		y_train_npy = npy.array(self.y_train)
		X_test_npy = npy.matrix(self.X_test)
		y_test_npy = npy.array(self.y_test)
		indices = [0, self.max_index, len(self.chords)]
		print indices
		with h5py.File("chorales.hdf5", "w", libver='latest') as f:
			f.create_dataset("X_train", (X_train_npy.shape[0], X_train_npy.shape[1]), dtype='i', data=X_train_npy)
			f.create_dataset("y_train", (y_train_npy.shape[0],), dtype='i', data=y_train_npy)
			f.create_dataset("X_test", (X_test_npy.shape[0], X_test_npy.shape[1]), dtype='i', data=X_test_npy)
			f.create_dataset("y_test", (y_test_npy.shape[0],), dtype='i', data=y_test_npy)
			f.create_dataset("indices", (1, 3), dtype='i', data=indices)


	# Split the quantized scores into a training and test split
	def training_test_split(self, score_list):
		if not self.analyzed:
			raise Exception("Call analyze() first to get the quanitzed scores.")

		shuffle(score_list)
		num_scores = len(score_list)
		split_point = int(num_scores * self.percentage_test)
		self.training_split = score_list[split_point:]
		self.test_split = score_list[:split_point]
		print "Total scores: %d" % len(score_list)
		print "Training split size: %d" % len(self.training_split)
		print "Test split size: %d" % len(self.test_split)
		return self.training_split, self.test_split

	# Quantize a score
	def quantize_score(self, score):
		for voice in PARTS:
			quantize(score, voice)
		self.quantized.append(score)

	# Returns the pitch value for the input note
	def get_pitch(self, n):
		return n.midi

	# Return the beat strength value for the input note
	def get_beat_str(self, n):
		return int(math.floor(n.beat))

	# Returns 1 if the input note is at a cadence, else 0
	def get_iscadence(self, n):
		return 1 if hasFermata(n) else 0

	# Returns the input note's distance to the next cadence
	def get_cadence_dist(self, n, index, notes_lst, fermata_locations):
		if index == len(notes_lst) - 1 or hasFermata(n):
			return 0
		return fermata_locations[index:].index(True)

	# Returns the input note's distance to the end of the chorale 
	def get_offset_end(self, n, last_note):
		return math.floor(last_note.offset) - math.floor(n.offset)

	# Returns a tuple of midi values representing the harmony for the soprano note at index i
	def get_chord(self, i, a, t, b):
		return a[i].midi, t[i].midi, b[i].midi

	def __str__(self):
		s = "\n---------- FEATURIZER RESULTS ----------\n"
		for feature, lst in self.ordering.iteritems():
			s += feature + ": " + str(lst) + "\n"
		s += "INDICES: %s\n" % str(self.indices)
		s += "CHORD INDICES: 1 to %d [example chord: %s]\n" % (len(self.chords), str(list(self.chords)[0]))
		s += "Test-training split: %d training chorales, %d test chorales\n" % (len(self.training_split), len(self.test_split))
		s += "Test-training examples: %d for training, %d for test\n" % (len(self.X_train), len(self.X_test))
		s += "---------------------------------------\n"
		return s

	__repr__ = __str__

	

#from wrangle1 import *
sf = SopranoFeaturizer(40)
sf.analyze()
sf.featurize()
sf.verify()
sf.write()
print sf


