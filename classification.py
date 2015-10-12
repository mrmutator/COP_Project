import classifier as cl
from par2vec import load_all_models
from corpora import get_utterances_from_file
from gensim.models import Doc2Vec
from swda import CorpusReader
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import re
import numpy as np
import sys

def load_data():
	train_utt = []
	train_Y = []
	test_utt = []
	test_Y = []
	for tag, tokens in get_utterances_from_file("data/swda_utterances.train"):
		train_utt.append(" ".join(tokens))
		# remove id from tag
		tag = tag.split("/")[0]
		train_Y.append(tag)
	for tag, tokens in get_utterances_from_file("data/swda_utterances.test"):
		test_utt.append(" ".join(tokens))
		# remove id from tag
		tag = tag.split("/")[0]
		test_Y.append(tag)

	return train_utt, train_Y, test_utt, test_Y

def encode_tags(train_Y, test_Y):
	le = preprocessing.LabelEncoder()
	le.fit(train_Y+test_Y)

	train_Y = le.transform(train_Y) # return normalized tags
	test_Y = le.transform(test_Y)
	return train_Y, test_Y

def represent_random(train_utt, test_utt):   	
	# this doesn't actually do anything meaningfull
	train_X = []
	test_X = []
	i = 0
	for trans in test_utt:
		for utt in trans:
			test_X.append(np.random.random(10))
			sys.stderr.write("\r") ; sys.stderr.write("utterance %s" % i) ; sys.stderr.flush(); i += 1
	for trans in train_utt:
		for utt in trans:
			train_X.append(np.random.random(10))
			sys.stderr.write("\r") ; sys.stderr.write("utterance %s" % i) ; sys.stderr.flush(); i += 1
	sys.stderr.write("\n") 
	return train_X, test_X

def represent_simple(train_utt, test_utt, model):   	
	# this represents every utterance as it's own embedding this should be extended to encorporate some context
	train_X = []
	test_X = []
	i = 0
	for utt in test_utt:
		test_X.append(model.infer_vector(utt.split()))
		sys.stderr.write("\r") ; sys.stderr.write("utterance %s" % i) ; sys.stderr.flush(); i += 1
	for utt in train_utt:
		train_X.append(model.infer_vector(utt.split()))
		sys.stderr.write("\r") ; sys.stderr.write("utterance %s" % i) ; sys.stderr.flush(); i += 1
	sys.stderr.write("\n") 
	return train_X, test_X

def bow_representation(train_utt, test_utt):
	# how are we going to do some context representation here?
	vectorizer = CountVectorizer(min_df = 1)
	# fit + transform on all training data
	train_X = vectorizer.fit_transform(train_utt)
	# only transform on all test data
	test_X = vectorizer.transform(test_utt)

	return train_X, test_X

def baseline_scores(train_utt, train_Y, test_utt, test_Y ):
	print "Calculating baseline BOW scores"
	# Baseline scores, use BOW representation of utterances
	train_X, test_X = bow_representation(train_utt, test_utt)
	print "BOW representation created"
	print "KNN Accuracy: ",  cl.KNN_classifier(train_X, train_Y, test_X, test_Y)
	# print "SVM Accuracy: ", cl.SVM_classifier(train_X, train_Y, test_X, test_Y)
	print "NB Accuracy: ", cl.NB_classifier(train_X, train_Y, test_X, test_Y)
	


if __name__ == '__main__':
	# load training and test data
	train_utt, train_Y, test_utt, test_Y = load_data()
	# encode tags
	train_Y, test_Y = encode_tags(train_Y, test_Y)

	# uncomment this to find basline scores
	# baseline_scores(train_utt, train_Y, test_utt, test_Y)

	print "Creating representations"
	# Load utterance embedding models
	embedding_model_location = "data/only_train_swda" #location of the embeddings
	embedding_model, _ = load_all_models(embedding_model_location)

	# represent utterances in some way, 
	train_X, test_X = represent_simple(train_utt, test_utt, embedding_model)
	
	# print np.array(train_X).shape, np.array(test_X).shape, np.array(train_Y).shape, np.array(test_Y).shape
	print "Training classifiers"
	# Train classifiers, print scores
	print "KNN Accuracy: ", cl.KNN_classifier(train_X, train_Y, test_X, test_Y)
	# print "SVM Accuracy: ", cl.SVM_classifier(train_X, train_Y, test_X, test_Y)
	print "NB Accuracy: ", cl.NB_classifier(train_X, train_Y, test_X, test_Y)