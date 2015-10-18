from music21 import *
import math
import os
from sys import maxsize
from pprint import pprint as pp
import numpy as npy

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
# 	2) beat strength [1 unit (4, 2, 3, 1)] ASK ABOUT THIS
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


# Create a binary vector to represent a MIDI note for a given voice range
# <n>: a midi note, where m_min <= n.midi <= m_max
# <m_min>: the lowest midi note for the given part
# <m_max>: the highest midi note for the given part
def vectorizeMIDI(n, m_min, m_max):
	vector = [VECTOR_OFF for i in range(m_max - m_min + 1)]
	vector[n.midi - m_min] = VECTOR_ON
	assert any(x == VECTOR_ON for x in vector)
	return vector

# Create a vector for the harmony voices(Alto, Tenor, and Bass)
# <alto>: the Alto note in MIDI
# <tenor>: the Tenor note in MIDI
# <bass>: the Bass note in MIDI
def vectorizeHarmonyNotes(alto, tenor, bass):
	alto_v = vectorizeMIDI(alto, RANGE['Alto']['min'], RANGE['Alto']['max'])
	tenor_v = vectorizeMIDI(tenor, RANGE['Tenor']['min'], RANGE['Tenor']['max'])
	bass_v = vectorizeMIDI(bass, RANGE['Bass']['min'], RANGE['Bass']['max'])
	return alto_v + tenor_v + bass_v


# Iterates over each example in the chorale and return a list of lists, where 
# each list represents the input features of a single soprano note.
# <soprano>: the soprano music21.stream.Part
def generateChoraleInputVectors(soprano):
	vectors = []
	time_signature = getTimeSignature(soprano)
	key_signature = getKeySignature(soprano)
	notes_lst = getNotes(soprano)
	fermata_locations = map(hasFermata, notes_lst)

	for n in notes_lst:
		index = notes_lst.index(n)
		v = []

		# Represent pitch as a binary vector [22 units] (range of soprano)
		pitch_v = indexSoprano([n])
		v += pitch_v

		# Represent beat strength as 1-based beat position [1 unit]
		beat_strength_v = [ math.floor(n.beat) ]
		assert beat_strength_v[0] % 1 == 0
		v += beat_strength_v

		# Represent cadence (contains a fermata) as a boolean [1 unit]
		cadence_v = [ VECTOR_ON if hasFermata(n) else VECTOR_OFF ]
		v += cadence_v

		# Represent distance to the next fermata [1 unit] (1 = on fermata)
		if index == len(notes_lst) - 1 or hasFermata(n):
			cadence_dist_v = [ VECTOR_OFF ]
		else:
			cadence_dist_v = [ fermata_locations[index + 1:].index(True) + VECTOR_ON ]
		v += cadence_dist_v

		# Represent offset from the beginning of the work [1 unit]
		# TODO: should be adjusted for pickups
		offset_start_v = [ math.floor(n.offset) ]
		v += offset_start_v

		# Represent offset from the end of the work [1 unit]
		offset_end_v = [ len(notes_lst) - 1 - math.floor(n.offset) ]
		v += offset_end_v

		# Represent time signature [2 units]
		time_sig_v = [ time_signature.numerator, time_signature.denominator ]
		v += time_sig_v

		# Represent key signature (sharps/flats, major/minor) [2 units]
		key_sig_v = [ key_signature.sharps, VECTOR_ON if key_signature.mode == 'major' else VECTOR_OFF ]
		v += key_sig_v
		
		# Total: 31 input units
		#print pitch_v
		#print beat_strength_v
		#print cadence_v
		#print cadence_dist_v
		#print offset_start_v
		#print offset_end_v
		#print time_sig_v
		#print key_sig_v
		vectors.append(v)
	return vectors


def generateChoraleOutputVectors(alto, tenor, bass):
	vectors = []
	alto_notes = alto.flat.notes
	tenor_notes = tenor.flat.notes
	bass_notes = bass.flat.notes
	for i in range(len(alto_notes)):
		vectors.append(vectorizeHarmonyNotes(alto_notes[i], tenor_notes[i], bass_notes[i]))
	return vectors


def indexSoprano(soprano):
	s_range = range(RANGE['Soprano']['min'], RANGE['Soprano']['max'] + 1)
	return map(lambda n: s_range.index(n.midi), soprano)

# Returns a list of unique numbers, each representing a tuple of an alto, tenor, and bass note
def indexAltoTenorBass(a, t, b):
	a_range = RANGE['Alto']['max'] - RANGE['Alto']['min'] + 1
	t_range = RANGE['Tenor']['max'] - RANGE['Tenor']['min'] + 1
	b_range = RANGE['Bass']['max'] - RANGE['Bass']['min'] + 1
	base = max(a_range, t_range, b_range) # luckily, the base is a prime (29)
	return map(lambda i: a[i].midi * (base**2) + t[i].midi * base + b[i].midi, range(len(a)))


def findUniqueATB():
	global QUANTIZED
	voicing_dict = {}
	iterator = corpus.chorales.Iterator(1, 371, numberingSystem = 'riemenschneider')
	index = 1
	for score in iterator[:20]:
		# TODO: find a better solution to this? (#11, 38, etc.)
		if len(score.parts) != 4:
			print "#%d: %s - TOO MANY PARTS (%s)" % (index, score.metadata.title, len(score.parts))
			index += 1
			continue
		# quantize
		print "#" + str(index) + ": Quantizing score " + score.metadata.title + ": ",
		for voice in PARTS:
			print voice[0],
			quantize(score, voice)
		print
		QUANTIZED.append(score)

		# gather voicings
		a = getNotes(score.parts[1])
		t = getNotes(score.parts[2])
		b = getNotes(score.parts[3])
		voicing_lst = indexAltoTenorBass(a, t, b) # returns an "ID"
		for i in range(len(a)):
			if voicing_lst[i] not in voicing_dict:
				voicing_dict[voicing_lst[i]] = a[i].midi, t[i].midi, b[i].midi
			# Check that no duplicate ID occurs for a different chord
			# This ensures every chord has a unique ID
			else:
				assert voicing_dict[voicing_lst[i]] == (a[i].midi, t[i].midi, b[i].midi)
		index += 1
	return voicing_dict


# Returns input matrix and output vector
def wrangleChorale(score, keys):
	# Check quantizing worked
	soprano = score.parts[0]
	alto = score.parts[1]
	tenor = score.parts[2]
	bass = score.parts[3]
	assert len(getNotes(soprano)) == len(getNotes(alto)) == len(getNotes(tenor)) == len(getNotes(bass))

	# Represent pitch as a binary vector [22 units] (range of soprano)
	# pitch_idx = indexSoprano(getNotes(soprano))
	# npy_pitch_idx = npy.array(pitch_idx)
	# print npy_pitch_idx
	vectors = generateChoraleInputVectors(soprano)
	npy_input_matrix = npy.matrix(vectors)

	# Represent output values (using primes)
	output_idx = indexAltoTenorBass(getNotes(alto), getNotes(tenor), getNotes(bass))
	output_idx = map(lambda n: keys.index(n), output_idx)
	npy_output_vec = npy.array(output_idx)

	return npy_input_matrix, npy_output_vec


def run():
	# Preprocess and determine possible voicings
	print "Finding voicings"
	VOICINGS = findUniqueATB()
	keys = VOICINGS.keys()
	print "VOICINGS size: %d" % len(VOICINGS)
	X = None # final input matrix
	y = None # final output vector
	for score in QUANTIZED:
		input_matrix, output_vector = wrangleChorale(score, keys)
		X = input_matrix if X is None else npy.vstack((X, input_matrix))
		y = output_vector if y is None else npy.hstack((y, output_vector))

	print X.shape
	print y.shape


#run()

iterator = corpus.chorales.Iterator(1, 371, numberingSystem = 'riemenschneider')
index = 1
for chorale in iterator[:30]:
	print index, chorale.metadata.title
	index += 1

	



















