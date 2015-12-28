#########
#
## File: clean.py
## Author: Hugh Zabriskie (c) 2015
## Description: Clean and quantize the Bach chorales.
#
#
# Total 4-voice chorales cleaned: 326
#
# TODO: write a script to go through all of the chorales and find any where key_sig.mode == None
#
#########

from helpers import *


# Remove all slurs from a part
def removeTies(_score, _partID):
	part = getPart(_score, _partID)
	for ms in getMeasures(part):
		for n in ms:
			if isNote(n):
				n.tie = None

# Remove rests from a stream by lengthening note that precede it
# Note that this function recurses upon finding a rest. _stream.remove() causes indexing errors
# if 'nar' is not reset.
def removeRests(_stream):
	nar, repeat = _stream.notesAndRests, False
	for nr in nar:
		if isRest(nr):
			prev_note = nar[nar.index(nr) - 1]
			prev_note.quarterLength += nr.quarterLength
			_stream.remove(nr)
			repeat = True
			break
	if repeat:
		return removeRests(_stream)
	return _stream

# If a chord n is found in a measure ms, replace it with the top note in the chord
def replaceChordWithNote(n, ms):
	top_note = note.Note(n.pitches[-1].nameWithOctave, quarterLength=n.duration.quarterLength)
	ms.insert(n.offset, top_note)
	ms.remove(n)
	return top_note


# If there is no fermata in the last measure, add one to the last beat (and any
# previous beats that have the same harmony - to cover where the last note was longer than 1 beat)
def addFinalFermatas(score):
	s, a, t, b = [getPart(score, x) for x in PARTS] # get parts
	#  list of last measures, each a list of notes, but reversed
	slm, alm, tlm, blm = [getMeasures(x)[-1].notes for x in [s, a, t, b]]

	# Add fermata to the last note
	addFermata([slm[-1], alm[-1], tlm[-1], blm[-1]])

	# Work backwards (the note lists are reversed)
	for i in range(len(slm) - 1)[::-1]:
		if slm[i].midi == slm[-1].midi and alm[i].midi == alm[-1].midi \
		and tlm[i].midi == tlm[-1].midi and blm[i].midi == blm[-1].midi:
			addFermata([slm[i], alm[i], tlm[i], blm[i]])


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
				n = replaceChordWithNote(n, ms)
				notes = getNotes(ms)

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
					if isChord(next_note):
						replaceChordWithNote(next_note, ms)
					context.append(next_note)
					context_duration += length(next_note)
				if context_duration == 1.0:
					n.duration.quarterLength = 1.0
					for other_note in context[1:]:
						ms.remove(other_note)
				# keep the 1st and last note
				elif context_duration >= 2.0:
					# make the last k notes into quarter notes
					for n in context[-int(context_duration):]:
						n.duration.quarterLength = 1.0
					map(lambda x : ms.remove(x), context[:-int(context_duration)])


			# break down half notes, dotted halves, whole notes (keeping fermatas)
			elif length(n) > 1 and length(n) % 1 == 0:
				total_beats = int(length(n))
				n.quarterLength = 1.0
				for beat in range(1, total_beats):
					new_note = note.Note(n.pitch.nameWithOctave, quarterLength=1.0)
					new_note.expressions = [expressions.Fermata()] if hasFermata(n) else []
					# avoid unneccessary accidentals
					ms.insert(n.offset + beat, new_note)

			# dotted quarter
			elif length(n) > 1:
				next_note = notes[index + 1]
				# dotted quarter then a quarter note
				if length(next_note) >= 1:
					next_note.duration.quarterLength += n.duration.quarterLength - 1.0 # propagate the error forward
					n.duration.quarterLength = 1.0
				# dotted quarter than 1 8th or 2 16ths
				else:
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



#### TEST ####

def test_chorale(name):
	s = corpus.parse(name)
	# Quarter-note quantization
	for voice in PARTS:
		quantize(s, voice)

	s.show()

# test_chorale('bach/bwv371')

def clean_chorale(score, index):
	# Quarter-note quantization
	for voice in PARTS:
		quantize(score, voice)
	# Ensure fermatas at the end of the piece
	addFinalFermatas(score)
	print "Cleaned score #%d: %s" % (index, score.metadata.title)
	
	# Record metadata
	with open("raw_data/data.txt", "a") as f:
		f.write("%d.xml: %s\n" % (index, score.metadata.title))
	
	# Save them to a MusicXML file.
	with open('raw_data/%d.xml' % index, 'w') as f:
		f.write(musicxml.m21ToString.fromStream(score))

#### RUN ####

def run():
	iterator = corpus.chorales.Iterator(numberingSystem = 'riemenschneider')
	counter = 0
	for score in iterator:
		if len(score.parts) == 4:
			counter += 1
			clean_chorale(score, counter)
			
	print "--"*20
	print "Gathered %d 4-part chorales" % counter
	print "--"*20

run()


