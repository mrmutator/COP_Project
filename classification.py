import classifier as cl
from par2vec import load_all_models
from gensim.models import Doc2Vec
from swda import CorpusReader
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import re
import numpy as np
import sys

def load_data(data_location, test_ids, debug = False):
	print "Reading data"
	# this loads the utterances and their tags from the swda corpus split in train and test set
	# where the ids of the test dialogs are listed in test_ids
	# utterances are list of transcripts, which are lists of utterances (so we can use compisitionality)
	corpus = CorpusReader(data_location)
	train_utt = []
	train_Y = []
	test_utt = []
	test_Y = []
	for trans in corpus.iter_transcripts(display_progress = True):
		if trans.conversation_no in test_ids:
			trans_utt = []
			# this is a test instance, put it in the test lists
			for utt in trans.utterances:
				# If the regex changes in preprocessing, also change it here!
				utt_tokens =  word_tokenize(re.sub(r'\{.+? |\}|\[|\]|\+|#|/|<.+?>', "", utt.text.lower())) 
				if utt_tokens:
					test_Y.append(utt.damsl_act_tag())
					trans_utt.append(utt_tokens)
			test_utt.append(trans_utt)
		else:
			trans_utt = []
			# this is a training instance, put it in the test lists
			for utt in trans.utterances:
				# If the regex changes in preprocessing, also change it here!
				utt_tokens =  word_tokenize(re.sub(r'\{.+? |\}|\[|\]|\+|#|/|<.+?>', "", utt.text.lower())) 
				if utt_tokens:
					train_Y.append(utt.damsl_act_tag())
					trans_utt.append(utt_tokens)
			train_utt.append(trans_utt)
		if debug == True and len(train_Y) > 10 and len(test_Y)> 10:
			break
	print "Data reading complete"
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
	# this represents every utterance as it's own embedding
	train_X = []
	test_X = []
	i = 0
	for trans in test_utt:
		for utt in trans:
			test_X.append(model.infer_vector(utt))
			sys.stderr.write("\r") ; sys.stderr.write("utterance %s" % i) ; sys.stderr.flush(); i += 1
	for trans in train_utt:
		for utt in trans:
			train_X.append(model.infer_vector(utt))
			sys.stderr.write("\r") ; sys.stderr.write("utterance %s" % i) ; sys.stderr.flush(); i += 1
	sys.stderr.write("\n") 
	return train_X, test_X

def bow_representation(train_utt, test_utt):
	# how are we going to do some context representation here?
	vectorizer = CountVectorizer(min_df = 1)
	# fit + transform on all training data
	train = []
	for trans in train_utt:
		for utt in trans:
			train.append(" ".join(utt))
	train_X = vectorizer.fit_transform(train)
	# only transform on all test data
	test = []
	for trans in test_utt:
		for utt in trans:
			test.append(" ".join(utt))
	test_X = vectorizer.transform(test)

	return train_X, test_X

def baseline_scores(train_utt, train_Y, test_utt, test_Y ):
	print "Calculating baseline BOW scores"
	# Baseline scores, use BOW representation of utterances
	train_X, test_X = bow_representation(train_utt, test_utt)
	knn_score = cl.KNN_classifier(train_X, train_Y, test_X, test_Y)
	svm_score = cl.SVM_classifier(train_X, train_Y, test_X, test_Y)
	nb_score = cl.NB_classifier(train_X, train_Y, test_X, test_Y)
	
	# print scores
	print "KNN Accuracy: ", knn_score
	print "SVM Accuracy: ", svm_score
	print "NB Accuracy: ", nb_score
	print "Creating representations"

if __name__ == '__main__':
	data_location = "data/swda"
	# list of ids that are in the test set, maybe should load from file
	test_ids = [2175,2053, 3360, 3389, 3926, 4078, 3054, 3852, 2708, 2121, 2562, 3745, 3254, 2455, 2749, 2330, 3207, 2505, 3495] 

	# load training and test data
	train_utt, train_Y, test_utt, test_Y = load_data(data_location, test_ids)

	# encode tags
	train_Y, test_Y = encode_tags(train_Y, test_Y)

	# uncomment this to find basline scores
	# baseline_scores(train_utt, train_Y, test_utt, test_Y)

	# Load utterance embedding models
	embedding_model_location = "data/test" #location of the embeddings

	embedding_model, _ = load_all_models(embedding_model_location)
	# represent utterances in some way, 
	train_X, test_X = represent_simple(train_utt, test_utt, embedding_model)
	
	# print np.array(train_X).shape, np.array(test_X).shape, np.array(train_Y).shape, np.array(test_Y).shape
	print "Training classifiers"
	# Train classifiers
	knn_score = cl.KNN_classifier(train_X, train_Y, test_X, test_Y)
	svm_score = cl.SVM_classifier(train_X, train_Y, test_X, test_Y)
	nb_score = cl.NB_classifier(train_X, train_Y, test_X, test_Y)
	
	# print scores
	print "KNN Accuracy: ", knn_score
	print "SVM Accuracy: ", svm_score
	print "NB Accuracy: ", nb_score