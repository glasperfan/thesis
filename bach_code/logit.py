from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import h5py
import numpy
from collections import Counter


def encode(train, test):
	encoder = OneHotEncoder()
	encoder.fit(numpy.vstack((train, test)))
	trainencoded = encoder.transform(train)
	testencoded = encoder.transform(test)
	return trainencoded, testencoded

def runLogitAndNB(Xtrainsparse, Xtestsparse):
	for i in range(len(ytrainraw[0])):
		print "Output type %i" % i
		logit1 = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1)
		logit2 = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', C=100)
		logit3 = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10000)
		nb1 = naive_bayes.MultinomialNB(alpha=0.01, fit_prior=True, class_prior=None)
		nb2 = naive_bayes.MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)
		nb3 = naive_bayes.MultinomialNB(alpha=1, fit_prior=True, class_prior=None)
		RF1 = RandomForestClassifier(1, "entropy", None)
		RF2 = RandomForestClassifier(10, "entropy", None)
		RF3 = RandomForestClassifier(20, "entropy", None)
		ytrain = numpy.hstack((ytrainraw[:, i], ydevraw[:, i]))
		ytest = ytestraw[:, i]
		RF1.fit(Xtrainsparse, ytrain)
		RF2.fit(Xtrainsparse, ytrain)
		RF3.fit(Xtrainsparse, ytrain)
		scores = [RF1.score(Xtestsparse, ytest), RF2.score(Xtestsparse, ytest), RF3.score(Xtestsparse, ytest)]
		print "R-FOREST: Best score %.2f%%, min of %.2f%%" % (max(scores) * 100, min(scores) * 100)
		ERF = ExtraTreesClassifier(n_estimators=40, max_depth=None, min_samples_split=1, random_state=0)
		ERF.fit(Xtrainsparse, ytrain)
		print "EXTRA TREES: Best score %.2f%%" % (ERF.score(Xtestsparse, ytest) * 100)
		nb1.fit(Xtrainsparse, ytrain)
		nb2.fit(Xtrainsparse, ytrain)
		nb3.fit(Xtrainsparse, ytrain)
		scores = [nb1.score(Xtestsparse, ytest), nb2.score(Xtestsparse, ytest), nb3.score(Xtestsparse, ytest)]
		print "MULTI-NB: Best score %.2f%%" % (max(scores) * 100)
		logit1.fit(Xtrainsparse, ytrain)
		logit2.fit(Xtrainsparse, ytrain)
		logit3.fit(Xtrainsparse, ytrain)
		scores = [logit1.score(Xtestsparse, ytest), logit2.score(Xtestsparse, ytest), logit3.score(Xtestsparse, ytest)]
		print "LOGIT: Best score %.2f%%" % (max(scores) * 100)
		most_common = lambda lst : max(set(list(lst)), key=list(lst).count)
		print "Most common class frequency: %.1f%% (train) %.1f%% (test)" % \
					(Counter(ytrain)[most_common(ytrain)] / float(len(ytrain)) * 100., \
					Counter(ytest)[most_common(ytest)] / float(len(ytest)) * 100.)
		print

# Load data
with h5py.File('data/chorales.hdf5', "r", libver='latest') as f:
    Xtrainraw = f['Xtrain'].value
    ytrainraw = f['ytrain'].value
    Xdevraw = f['Xdev'].value
    ydevraw = f['ydev'].value
    Xtestraw = f['Xtest'].value
    ytestraw = f['ytest'].value

# Cycle through the output targets and see how logistic regression performs based solely on the melody
def test1():
	print("1. Testing learning on melody alone...")
	Xtrain = numpy.vstack((Xtrainraw[:, range(0,10)], Xdevraw[:, range(0,10)]))
	Xtest = Xtestraw[:, range(0,10)]
	Xtrainsparse, Xtestsparse = encode(Xtrain, Xtest)
	runLogitAndNB(Xtrainsparse, Xtestsparse)


# Oracle experiments
def test2():
	print("2. Performing oracle experiment...")
	Xtrain = numpy.vstack((Xtrainraw, Xdevraw))
	Xtest = Xtestraw
	Xtrainsparse, Xtestsparse = encode(Xtrain, Xtest)
	runLogitAndNB(Xtrainsparse, Xtestsparse)

# Full harmonization
def test3():
	print("3. Testing softmax for full harmonization...")
	Xtrain = numpy.vstack((Xtrainraw, Xdevraw))
	ytrain = numpy.vstack((ytrainraw, ydevraw))
	Xtest = Xtestraw
	ytest = ytestraw
	Xall = numpy.vstack((Xtrain, Xtest))
	yall = numpy.vstack((ytrain, ytest))
	# Create the map between features and indices
	harmdict = {}
	for idx, feature in enumerate(list(set(map(tuple, yall)))):
		harmdict[feature] = idx + 1

	f = lambda x: harmdict[tuple(x)]
	ytrainfeat = numpy.array(map(f, ytrain))
	ytestfeat = numpy.array(map(f, ytest))
	yallfeat = numpy.array(map(f, yall))
	for idx in range(len(ytestfeat)):
		if ytestfeat[idx] not in ytrainfeat:
			ytestfeat[idx] = max(yallfeat) + 1
	yallfeat = numpy.hstack((ytrainfeat, ytestfeat))
	with h5py.File('data/chorales_sm.hdf5', "w", libver='latest') as f:
		f.create_dataset('ytrainfeat', ytrainfeat.shape, dtype="i", data=ytrainfeat)
		f.create_dataset('ytestfeat', ytestfeat.shape, dtype="i", data=ytestfeat)
		f.create_dataset('yallfeat', yallfeat.shape, dtype="i", data=yallfeat)
	print "data/chorales_sm.hdf5 written"
	Xtrainsparse, Xtestsparse = encode(Xtrain, Xtest)
	RF = RandomForestClassifier(10, "entropy", None)
	RF.fit(Xtrain, ytrainfeat)
	score_RF = RF.score(Xtest, ytestfeat)
	print "R-FOREST: score %.2f%%" % (score_RF * 100)
	ERF = ExtraTreesClassifier(n_estimators=40, max_depth=None, min_samples_split=1, random_state=0)
	ERF.fit(Xtrain, ytrainfeat)
	score_ERF = ERF.score(Xtest, ytestfeat)
	print "EXTRA TREES: score %.2f%%" % (score_ERF * 100)
	logit = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', C=100)
	logit.fit(Xtrain, ytrainfeat)
	score_logit = logit.score(Xtest, ytestfeat)
	print "LOGIT: score %.2f%%" % (score_logit * 100)

# test1()
# test2()
test3()


