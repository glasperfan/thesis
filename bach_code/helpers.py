#########
#
## File: helpers.py
## Author: Hugh Zabriskie (c) 2015
## Description: Useful constants and functions for manipulating music21 objects.
#
#########

from music21 import *

## CONSTANTS ##

# The 4 voices (and Part IDs) in a Bach chorale
PARTS = ['Soprano', 'Alto', 'Tenor', 'Bass']

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


## FUNCTIONS ##

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

def isRest(r):
	return isinstance(r, note.Rest)

def length(n):
	return n.duration.quarterLength

def hasFermata(n):
	return any(type(x) is expressions.Fermata for x in n.expressions)

def addFermata(notes):
	for n in notes:
		n.expressions = [expressions.Fermata()]