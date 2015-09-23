from music21 import *
import pprint


pp = pprint.PrettyPrinter(indent=4)
breakdown = {};
for i in range(0, 20):
	breakdown[i] = 0
works = corpus.getBachChorales()
for workName in works:
	work = converter.parse(workName)
	parts = len(work.getElementsByClass(stream.Part))
	breakdown[parts] = breakdown[parts] + 1
	if parts > 8:
		work.show()

pp.pprint(breakdown)
