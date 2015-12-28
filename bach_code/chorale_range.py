#########
#
## File: chorale_range.py
## Author: Hugh Zabriskie (c) 2015
## Description: Code to determine the range of voices .
#
#########

from music21 import *

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

# Given a dictionary for the set of chorales with the voice range for each voice.
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



### Correlated results with http://www.ofai.at/~soren.madsen/daimi/harmreport.pdf
# Soprano: 		[60; 81]	range: 22 		--all ranges inclusive
# Alto: 		[53; 74]	range: 22
# Tenor: 		[48; 69]	range: 22
# Bass: 		[36; 64]	range: 29

# Total range: 95 (inclusive)
