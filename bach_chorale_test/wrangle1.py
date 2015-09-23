from music21 import *
import math
import copy
from sys import maxsize
from pprint import pprint as pp

#####
#
# wrangle1.py
#
# Goal: create a data representation of the Bach chorale "Aus meines Herzens Grunde" (Riemenschneider 1, BWV 269) in
# order to build a training set for harmonizing the soprano voice.
# In other words, INPUT = soprano melody, OUTPUT = the 3 other voices
#
# For each INPUT note of the soprano melody, we extract the following features:
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
# The OUTPUT units are as follows: [3 units]
#	1) Alto note - distance from soprano tonic [1 unit]
#	2) Tenor note - distance from soprano tonic [1 unit]
#	3) Bass note - distance from soprano tonic [1 unit]
#
# TOTAL: 16 input units, 3 output units
#
# In the event that we don't have prior information (i.e. we are examining the first note), we set the previous 2 outputs to be all 0's.
#
# The goal is to provide accurate harmonization on each quarter note.
#####


# The 4 voices (and Part IDs) in a Bach chorale
PARTS = ['Soprano', 'Alto', 'Tenor', 'Bass']


def getTimeSignature(_stream):
	return _stream.flat.getElementsByClass(meter.TimeSignature)[0]

def getMeasures(_stream):
	return _stream.getElementsByClass(stream.Measure)

def getNotes(_stream):
	return _stream.getElementsByClass(note.Note)

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

def parse(_score):
	return corpus.parse(_score)

def removeTies(_score, _partID):
	part = getPart(_score, _partID)
	ts = getTimeSignature(part)

	for ms in getMeasures(part):
		for n in ms:
			if isNote(n):
				if n.tie is not None:
					n.tie = None


def quantize(_score, _partID):
	part = getPart(_score, _partID)
	ts = getTimeSignature(part)
	measures = getMeasures(part)

	# TODO: catch the 12/8 case
	if ts.ratioString is '12/8':
		return None

	# first, remove any ties
	removeTies(_score, _partID)

	# iterate over each measure
	for ms in measures:
		index = 0
		notes = getNotes(ms)

		# iterate over each note
		while index < len(notes):
			n = notes[index]

			# skip quarter notes
			if length(n) == 1:
				index += 1

			# combine notes that add up to one beat
			# 1/8G + 1/16A + 1/16B --> 1/4G
			elif length(n) < 1 and index < len(notes) - 1:
				context = [n]
				context_duration = length(n)
				while context_duration < 1:
					next_note = notes[index + len(context)]
					context.append(next_note)
					context_duration += length(next_note)
				if context_duration == 1:
					n.duration.quarterLength = 1.0
					for other_note in context[1:]:
						ms.remove(other_note)
				index += len(context)

			# break down half notes, dotted halves, whole notes
			elif length(n) > 1 and length(n) % 1 == 0:
				total_beats = int(length(n))
				n.quarterLength = 1.0
				for beat in range(1, total_beats):
					new_note = note.Note(n.pitch.nameWithOctave, quarterLength=1.0)
					# avoid unneccessary accidentals
					if n.accidental != None:
						new_note.accidental = None
					ms.insert(n.offset + beat, new_note)
				index += total_beats

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
				index += 2

	# part.show()


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


def showIndex(index):
	s = corpus.parse()

### Wrapper function: calls choraleRange on all chorales for all voices
### Results from http://www.ofai.at/~soren.madsen/daimi/harmreport.pdf CORRELATE


# Soprano: 		[60; 81]	range: 22 (inclusive)
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


# Loads into memory all 4-voice chorales as score objects
def loadChoraleScore():
	choralePaths = corpus.getBachChorales()
	chorales = []
	for path in choralePaths:
		s = corpus.parse(path)
		if len(s.parts) == 4:
			chorales.append(s)
	return chorales

##
#
# Data format:
# 
# Input units:
#	- previous chords
#	- soprano input (note + 3 previous + 3 next)
#
##
RANGE = {
	'Soprano': {
		'max': 81,
		'min': 60
	},
	'Alto': {
		'max': 74,
		'min': 53},
	},
	'Tenor': {d
		'max': 69,
		'min': 48
	},
	'Bass': {
		'max': 64,
		'min': 36
	}
}
TOTAL_RANGE = 95 # the sum of these ranges, inclusive


# Create a binary vector to represent a MIDI note for a given voice range
# <n>: a midi note, where m_min <= n <= m_max
# <m_min>: the lowest midi note for the given part
# <m_max>: the highest midi note for the given part
def vectorizeMIDI(n, m_min, m_max):
	vector = [0 for i in range(0, m_max - m_min + 1)]
	vector[n - m_min] = 1
	return vector

# Create a vector for the harmony voices(Alto, Tenor, and Bass)
# <alto>: the Alto note in MIDI
# <tenor>: the Tenor note in MIDI
# <bass>: the Bass notei in MIDI
def vectorizeHarmony(alto, tenor, bass):
	global RANGE
	return vectorizeMIDI(alto, RANGE.Alto.min, RANGE.Alto.max) +
		vectorizeMIDI(tenor, RANGE.Tenor.min, RANGE.Tenor.max) +
		vectorizeMIDI(bass, RANGE.Bass.min, RANGE.Bass.max)

# Normalizes the Soprano MIDI range. Therefore, 0 represents 'no note', 1 the lowest soprano note, and
# so on up to the highest note.
# <soprano
def normalizeSoprano(soprano):
	global RANGE
	return soprano - RANGE.Soprano.min + 1


def wrangleChorale(score):
	# Preprocessing
	for voice in PARTS:
		quantize(score, voice)
	#soprano = score.parts[0].notes
	#alto = score.parts[1].notes
	#tenor = score.parts[2].notes
	#bass = score.parts[3].notes
	#assert len(soprano) == len(alto) == len(tenor) == len(bass)

# iterator = corpus.chorales.Iterator()
# bwv269 = iterator.next()

# PREPROCESSING
# print "Preprocessing..."
# for voice in PARTS:
# 	quantize(bwv269, voice)

# Determine the global range
#allChoralesRange()



	



















