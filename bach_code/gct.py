#
#
# Implementation of the GCT Algorithm
# As outlined in Cambouropoulos et. al. 2014
# Available at: http://www.coinvent-project.eu/fileadmin/publications/icmc-smc2014-GCT-submitted.pdf
#
#

from music21 import chord
from music21 import pitch
from random import choice as rand_choice
from numpy import argmin

def dfs(row, consonance, subsets, recurse=False, prev=[]):
    # single element
    if len(row) < 2 and consonance[0]:
        if not recurse:
            subsets.append([row[0]])
        return [row[0]]
    # consonant pair
    subset = prev + [row[0]]
    for idx, el in enumerate(row[1:]):
        if consonance[el - row[0]]:
            subset += dfs(row[idx + 1:], consonance, subsets, True, subset)
    subsets.append(subset)
    return []

def find_consonant_subsets(pcs, consonance):
    subsets = []
    for idx in range(len(pcs)):
        row = pcs[idx:]
        dfs(row, consonance, subsets) # dfs fills subsets
    return subsets

def chord_to_intervals(ch):
    intervals = []
    for i in range(len(ch) - 1):
        intervals.append(ch[i + 1] - ch[i])
    return intervals

MAJOR = [0,4,7]
MINOR = [0,3,7]
DOMINANT = [0,4,7,10]

# If it's a major or minor triad, rotate the pcset to root position
def rotatetomode(pcs):
    all_rotations = [pcs[i:] + pcs[:i] for i in range(len(pcs))]
    for rotation in all_rotations:
        rooted = [(x - rotation[0]) % 12 for x in rotation]
        if rooted == MAJOR or rooted == MINOR or rooted == DOMINANT:
            return rotation
    return pcs

# Add non-base pcs as extensions
def addextensions(base, allpcs):
    extensionpcs = list(set(allpcs) - set(base))
    return base + [x + 12 for x in extensionpcs]

# Additional steps for GCT to ensure unique encodings
def optimalchoice(choices):
    # Dyads: select 5th over 4th, 7th over 2nd
    if len(choices[0][1]) == 2:
        for choice in choices:
            if sum(choice[1]) > 6: # 6 semitones
                return choice
    choices_f = [x for x in choices if x[1][1] < 9]
    if len(choices_f) < 1:
    	print choices
    for choice in choices_f:
        # Select dominant chords with the correct root
        if choice[1] == DOMINANT:
            return choice
    # Prefer chords with intervals larger than major 2nd (i.e. to select minor 7ths or major chords with add 6th)
    for choice in choices_f:
        intervals = chord_to_intervals(choice[1])
        if all(x > 2 for x in intervals):
            return choice
    return rand_choice(choices_f)


# GCT algorithm
# t = tonic pitch class
# v = binary consonance vector of length 12. v[i] denotes whether an interval of i semitones is consonant
# c = music21.chord.Chord
# OUTPUT = [root pc, bass pc, chord pcs], all relative to the tonic
def GCT(t, v, c):
    # Create pitch class set and remove duplicate pcs
    pcset_orig = list(set([x.midi % 12 for x in c.pitches])) 
    lowest_note = c.pitches[argmin([x.midi for x in c.pitches])]
    bass = (lowest_note.midi - t) % 12
    pcset = sorted(pcset_orig)
    # Find consonant subsets using DFS
    subsets = find_consonant_subsets(pcset, v)
    # Sort by length and grab the longest ones
    subsets.sort(key = len)
    maxsubsets = [x for x in subsets if len(x) == len(subsets[-1])] 
    rotated = map(rotatetomode, maxsubsets)
    # Special case: fully diminished chords
    for i in range(len(rotated)):
        ch = rotated[i]
        if all(x == 3 for x in chord_to_intervals(ch)):
            rotated += [ch[i:] + ch[:i] for i in range(len(ch))]
    # Add extensions and select root
    rotated = map(list, list(set(map(tuple,rotated))))
    withextensions = [addextensions(x, pcset) for x in rotated]
    withroot = [((x[0] - t) % 12, map(lambda e: (e - x[0]) % 12, x)) for x in withextensions]
    # Return the optimal choice to ensure unique encodings
    choice = withroot[0] if len(withroot) == 1 else optimalchoice(withroot)
    bass = (lowest_note.midi - t - choice[0]) % 12
    bass_idx = choice[1].index(bass)
    base = tuple(choice[1]) if len(choice[1]) > 1 else choice[1][0]
    result = choice[0], bass_idx, base
    return result



# Testing
def loop(func):
	for i in range(10):
		func()

def test():
    gmajor = 7
    cmajor = 0
    dmajor = 2
    v = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0] # Consonance vector
    c1 = chord.Chord([60, 62, 66, 69, 74])
    c2 = chord.Chord([50, 60, 62, 65, 69])
    c3 = chord.Chord([62, 68, 77, 71])
    c4 = chord.Chord(['F#3', 'D4', 'A4'])
    c5 = chord.Chord(['E3', 'G3', 'G4', 'C#5'])
    c6 = chord.Chord(['D3', 'A3', 'F#4', 'D4'])
    c7 = chord.Chord(['G3', 'B3', 'E4', 'E5'])
    c8 = chord.Chord(['A3', 'C#4', 'G4', 'E5'])
    c9 = chord.Chord(['D3', 'A3', 'F#4', 'D5'])
    c10 = chord.Chord(['G3', 'C4', 'G4', 'C5'])
    c11 = chord.Chord(['C3', 'D4'])
    c12 = chord.Chord(['D3', 'D4'])
    c13 = chord.Chord(['C4', 'C#4', 'D4', 'E-4'])
    c14 = chord.Chord(['C3', 'C4'])
    c15 = chord.Chord(['A-3', 'C4', 'E-4', 'G-4'])
    c16 = chord.Chord(['A-5', 'C4', 'E-4', 'G-4'])
    c17 = chord.Chord(['A-4', 'C5', 'E-4', 'G-4'])
    c18 = chord.Chord(['A-4', 'C5', 'E-5', 'G-4'])
    assert GCT(gmajor, v, c1) == (7, 3, (0, 4, 7, 10))
    assert GCT(cmajor, v, c2) == (2, 0, (0, 3, 7, 10))
    assert GCT(cmajor, v, c3) == (11, 1, (0, 3, 6, 9))
    assert GCT(dmajor, v, c4) == (0, 1, (0, 4, 7))
    assert GCT(dmajor, v, c5) == (11, 1, (0, 3, 6))
    assert GCT(dmajor, v, c6) == (0, 0, (0, 4, 7))
    assert GCT(dmajor, v, c7) == (2, 1, (0, 3, 7))
    assert GCT(dmajor, v, c8) == (7, 0, (0, 4, 7, 10))
    assert GCT(dmajor, v, c9) == (0, 0, (0, 4, 7))
    assert GCT(cmajor, v, c10) == (0, 1, (0, 7)) # Should choose a fifth (0,7)
    assert GCT(cmajor, v, c11) == (2, 1, (0, 10)) # Should choose a minor seventh (0, 10)
    assert GCT(dmajor, v, c12) == (0, 0, 0) # Should choose unison
    assert GCT(cmajor, v, c13) == (0, 0, (0, 3, 1, 2)) # Should choose minor third with strange extensions
    assert GCT(cmajor, v, c14) == (0, 0, 0)
    assert GCT(cmajor, v, c15) == (8, 0, (0, 4, 7, 10))
    assert GCT(cmajor, v, c16) == (8, 1, (0, 4, 7, 10))
    assert GCT(cmajor, v, c17) == (8, 2, (0, 4, 7, 10))
    assert GCT(cmajor, v, c18) == (8, 3, (0, 4, 7, 10))

#loop(test)
test()