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

def load_dataset(name, data_file):
	dataX, datay = [], []
	with h5py.File(data_file, "r", libver='latest') as f:
		counter = 0
		while True:
			try:
				dataX.append(f['%s/chorale%d_X' % (name, counter)].value)
				datay.append(f['%s/chorale%d_y' % (name, counter)].value)
			except:
				break
			counter += 1
	return dataX, datay

# Score without counting padding
def score_with_padding(pred, ytest, ypadding):
	correct = 0.0
	for idx, p in enumerate(pred):
		if ytest[idx] == p and ytest[idx] != ypadding:
			correct += 1
	return correct / ytest.shape[0]

# Full harmonization
def test3():
	print("3. Testing softmax for full harmonization...")
	trainXc, trainyc = load_dataset("train", "data/chorales_rnn.hdf5")
	devXc, devyc = load_dataset("dev", "data/chorales_rnn.hdf5")
	testXc, testyc = load_dataset("test", "data/chorales_rnn.hdf5")
	stack = lambda x1, x2: numpy.vstack((x1, x2))
	hstack = lambda x1, x2: numpy.hstack((x1, x2))
	Xtrain = stack(reduce(stack, trainXc), reduce(stack, devXc))
	ytrain = hstack(reduce(hstack, trainyc), reduce(hstack, devyc))
	Xtest, ytest = reduce(stack, testXc), reduce(hstack, testyc)

	ypadding = ytest.max()
	Xtrain_up, ytrain_up, Xtest_up, ytest_up = [], [], [], []
	for idx, p in enumerate(ytrain):
		if p != ypadding:
			Xtrain_up.append(Xtrain[idx])
			ytrain_up.append(ytrain[idx])
	for idx, p in enumerate(ytest):
		if p != ypadding:
			Xtest_up.append(Xtest[idx])
			ytest_up.append(ytest[idx])
	Xtrain, ytrain, Xtest, ytest = numpy.array(Xtrain_up), numpy.array(ytrain_up), \
								   numpy.array(Xtest_up), numpy.array(ytest_up)

	Xtrainsparse, Xtestsparse = encode(Xtrain, Xtest)
	RF = RandomForestClassifier(10, "entropy", None)
	RF.fit(Xtrain, ytrain)
	score_RF = score_with_padding(RF.predict(Xtest), ytest, ytest.max())
	print "R-FOREST: score %.2f%%" % (score_RF * 100)
	ERF = ExtraTreesClassifier(n_estimators=40, max_depth=None, min_samples_split=1, random_state=0)
	ERF.fit(Xtrain, ytrain)
	score_ERF = ERF.score(Xtest, ytest)
	print "EXTRA TREES: score %.2f%%" % (score_ERF * 100)
	logit = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1)
	logit.fit(Xtrain, ytrain)
	score_logit = logit.score(Xtest, ytest)
	print "LOGIT: score %.2f%%" % (score_logit * 100)

# test1()
# test2()
test3()


