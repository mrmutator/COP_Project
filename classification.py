import classifier as cl
from par2vec import load_all_models
from corpora import get_utterances_from_file
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import hidden_markov_model as hmm
import numpy as np
import sys
from itertools import groupby


def load_data_hmm():
    train_sequences = []
    test_sequences = []

    curr_sequence = []
    first = True
    for tag, tokens in get_utterances_from_file("data/swda_utterances.train"):
        if first: 
            cur_id = tag.split('/')[1].split('_')[0]
            first = False
        elif cur_id !=  tag.split('/')[1].split('_')[0]:
            cur_id = tag.split('/')[1].split('_')[0]
            train_sequences.append(curr_sequence)
            curr_sequence = []
        curr_sequence.append((tag, " ".join(tokens)))
    train_sequences.append(curr_sequence)
    curr_sequence = []
    first = True

    for tag, tokens in get_utterances_from_file("data/swda_utterances.test"):
        if first: 
            cur_id = tag.split('/')[1].split('_')[0]
            first = False
        elif cur_id !=  tag.split('/')[1].split('_')[0]:
            cur_id = tag.split('/')[1].split('_')[0]
            test_sequences.append(curr_sequence)
            curr_sequence = []
        curr_sequence.append((tag, " ".join(tokens)))
    test_sequences.append(curr_sequence)
    curr_sequence = []

    return train_sequences, test_sequences

def load_data():
    train_utt = []
    train_Y = []
    test_utt = []
    test_Y = []
    for tag, tokens in get_utterances_from_file("data/swda_utterances.train"):
        train_utt.append(" ".join(tokens))
        train_Y.append(tag)
    for tag, tokens in get_utterances_from_file("data/swda_utterances.test"):
        test_utt.append(" ".join(tokens))
        test_Y.append(tag)

    return train_utt, train_Y, test_utt, test_Y


def encode_tags(train_Y, test_Y):
    train_Y = [t.split("/")[0] for t in train_Y]
    test_Y = [t.split("/")[0] for t in test_Y]
    le = preprocessing.LabelEncoder()
    le.fit(train_Y + test_Y)

    train_Y = le.transform(train_Y)  # return normalized tags
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
            sys.stderr.write("\r");
            sys.stderr.write("utterance %s" % i);
            sys.stderr.flush();
            i += 1
    for trans in train_utt:
        for utt in trans:
            train_X.append(np.random.random(10))
            sys.stderr.write("\r");
            sys.stderr.write("utterance %s" % i);
            sys.stderr.flush();
            i += 1
    sys.stderr.write("\n")
    return train_X, test_X


def represent_simple(utterances, model):
    # this represents every utterance as it's own embedding this should be extended to encorporate some context
    X = []
    i = 0
    for utt in utterances:
        X.append(model.infer_vector(utt.split()))
        sys.stderr.write("\r");
        sys.stderr.write("utterance %s" % i);
        sys.stderr.flush();
        i += 1
    sys.stderr.write("\n")
    return X

def represent_lookup(labels, model):
    # this represents every utterance as it's own embedding this should be extended to encorporate some context
    X = []
    i = 0
    for label in labels:
        X.append(model.docvecs[label])
        sys.stderr.write("\r");
        sys.stderr.write("utterance %s" % i);
        sys.stderr.flush();
        i += 1
    sys.stderr.write("\n")
    return X


def bow_representation(train_utt, test_utt):
    # how are we going to do some context representation here?
    vectorizer = CountVectorizer(min_df=1)
    # fit + transform on all training data
    train_X = vectorizer.fit_transform(train_utt)
    # only transform on all test data
    test_X = vectorizer.transform(test_utt)

    return train_X, test_X


def baseline_scores(train_utt, train_Y, test_utt, test_Y):
    print "Calculating baseline BOW scores"
    # Baseline scores, use BOW representation of utterances
    train_X, test_X = bow_representation(train_utt, test_utt)
    print "BOW representation created"
    print "KNN Accuracy: ", cl.KNN_classifier(train_X, train_Y, test_X, test_Y)
    # print "SVM Accuracy: ", cl.SVM_classifier(train_X, train_Y, test_X, test_Y)
    print "NB Accuracy: ", cl.NB_classifier(train_X, train_Y, test_X, test_Y)

def evaluate_model(embedding_model_location):
    # load training and test data
    train_utt, train_Y, test_utt, test_Y = load_data()

    # uncomment this to find basline scores
    # baseline_scores(train_utt, train_Y, test_utt, test_Y)

    print "Creating representations"
    # Load utterance embedding models
    embedding_model, _ = load_all_models(embedding_model_location)

    # represent utterances in some way,
    # train_X = represent_simple(train_utt, embedding_model)
    # test_X = represent_simple(test_utt, embedding_model)

    train_X = represent_lookup(train_Y, embedding_model)
    test_X = represent_simple(test_utt, embedding_model)


    # encode tags
    train_Y, test_Y = encode_tags(train_Y, test_Y)

    # print np.array(train_X).shape, np.array(test_X).shape, np.array(train_Y).shape, np.array(test_Y).shape
    print "Training classifiers"
    # Train classifiers, print scores
    print "Model: ", embedding_model_location
    print "MLP Accuracy: ", cl.MLP_classifier(train_X, train_Y, test_X, test_Y, n_iter=10)
    print "KNN Accuracy: ", cl.KNN_classifier(train_X, train_Y, test_X, test_Y)
    # print "SVM Accuracy: ", cl.SVM_classifier(train_X, train_Y, test_X, test_Y)
    print "NB Accuracy: ", cl.NB_classifier(train_X, train_Y, test_X, test_Y)

def classify_context_dependent():
    # load data as:
    # training:
        # sequences of DA tags with corresponding speaker sequences for discourse model
        # utterances with corresponding DA tags for language model
    # testing:
        # sequences of utterances with corresponding speakers and DA tags (for evaluation)
    print "Loading Data"
    train_dialogs, test_dialogs =  load_data_hmm()
    
    

    # train a label encoder
    all_tags = []
    i = 0
    for dialog in train_dialogs+test_dialogs:
        for tag, utterance in dialog:
            all_tags.append(tag.split('/')[0])
            i += 1
    le = preprocessing.LabelEncoder()
    transformed_tags = le.fit_transform(all_tags)

    # This is not the way to do it
    emmision_probabilites = np.array([len(list(group)) for key, group in groupby(sorted(transformed_tags))])/float(len(transformed_tags))


    print "Loading sentence model"
    # train/load 'language model'
    # load a doc2vec model to train the language model with
    # model = None will give a uniform chance for each tag and utterance
    model = emmision_probabilites


    # learn transition matrix
    speaker_sequences = []
    tag_sequences = []
    for dialog in train_dialogs:
        speaker_sequences.append([tag[0].split('/')[1].split('_')[1] for tag in dialog])
        tag_sequences.append(le.transform([tag[0].split('/')[0] for tag in dialog]))
    print "Learning a Transitioin Matrix"
    start_prob, transition_matrix = hmm.learn_transition_matrix(tag_sequences, speaker_sequences, number_of_states = 43, order = 2)
    # print start_prob, transition_matrix

    print "Decoding"
    # evauluation:
    correct = 0
    total = 0
    for dialog in test_dialogs:
        utterances = [words[1] for words in dialog] # this needs to be given the representation it needs in the language model
        true_tags = le.transform([tag[0].split('/')[0] for tag in dialog])
        speakers = [tag[0].split('/')[1].split('_')[1] for tag in dialog]
        predicted = hmm.viterbi_decoder(utterances,speakers, start_prob, transition_matrix,  model, order = 2)
        score = hmm.evaluate_sequence(predicted, true_tags)
        correct += score
        total += len(dialog)
    print float(correct)/total
        # for every testing dialog, decode with viterbi
    # calculate accuracy maybe confusion matrix


if __name__ == '__main__':
    classify_context_dependent()

