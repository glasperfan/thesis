from music21 import *
import math
import os
from sys import maxsize
import numpy as npy

# Iterates over each example in the chorale and return a list of lists, where 
# each list represents the input features of a single soprano note.
# <soprano>: the soprano music21.stream.Part

#
#  1  pitch  22 23 beat str  26 27  cadence?  28 29  cadence_dist   59 60  off_end  160 161 time  162 163  key  174
# |------------|----------------|---------------|---------------------|----------------|-------------|-------------|
#
def generateChoraleInputVectors(soprano):
	vectors = []
	time_signature = getTimeSignature(soprano)
	key_signature = getKeySignature(soprano)
	notes_lst = getNotes(soprano)
	fermata_locations = map(hasFermata, notes_lst)

	for index, n in enumerate(notes_lst):
		v = []

		# Represent pitch as a binary vector [22 units] (range of soprano)
		pitch_v = indexSoprano(n)
		assert pitch_v >= 1 and pitch_v <= 22
		v += pitch_v

		# Represent beat strength as 1-based beat position [1 unit]
		beat_strength_v = 22 + int(math.floor(n.beat))
		assert beat_strength_v >= 23 and beat_strength_v <= 26
		v += beat_strength_v

		# Represent cadence (contains a fermata) as a boolean [1 unit]
		cadence_v = 27 + [ VECTOR_ON if hasFermata(n) else VECTOR_OFF ]
		assert cadence_v == 27 or cadence_v == 28
		v += cadence_v

		# Represent distance to the next fermata [1 unit] (1 = on fermata)
		if index == len(notes_lst) - 1 or hasFermata(n):
			cadence_dist_v = 29
		else:
			cadence_dist_v = 30 + fermata_locations[index + 1:].index(True)
		assert cadence_dist_v >= 29 and cadence_dist_v <= 59
		v += cadence_dist_v

		# Represent offset from the beginning of the work [1 unit]
		# TODO: should be adjusted for pickups
		# offset_start_v = [ math.floor(n.offset) ]
		# v += offset_start_v

		# Represent offset from the end of the work [1 unit]
		offset_end_v = 60 + len(notes_lst) - 1 - math.floor(n.offset)
		#assert offset_end_v >= 60 and offset_end_v <= 
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
		v = pitch_v[0]
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

# Returns the index of the soprano note, [1 : 22]
def indexSoprano(soprano_note):
	s_range = range(RANGE['Soprano']['min'], RANGE['Soprano']['max'] + 1)
	return s_range.index(soprano_note.midi) + 1
	#return map(lambda n: s_range.index(n.midi), soprano)

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
	print vectors
	npy_input_matrix = npy.array(vectors)

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
		print input_matrix.shape
		X = input_matrix if X is None else npy.hstack((X, input_matrix))
		y = output_vector if y is None else npy.hstack((y, output_vector))

	print X.shape
	print y.shape

	npy.savetxt('X.txt', X, delimiter=',')
	npy.savetxt('y.txt', y, delimiter=',')