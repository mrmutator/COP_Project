import classifier as cl
from par2vec import load_all_models
from gensim.models import Doc2Vec
from corpora import get_utterances_from_file
from sklearn import preprocessing
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
import hidden_markov_model as hmm
import numpy as np
import sys
from itertools import groupby
from operator import add
from operator import concat
from itertools import islice
from corpora import get_swda_utterances
from tagset import get_aggregated_tagset_mapping
from utils import Doc2VecKeyError

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


def encode_tags(train_Y, test_Y, aggregated_tagset=False):

    mapping = get_aggregated_tagset_mapping()

    # get tags according to aggregated group
    if not aggregated_tagset:
        train_Y = [t.split("/")[0] for t in train_Y]
        test_Y = [t.split("/")[0] for t in test_Y]
    else:
        train_Y = [mapping[t.split("/")[0]] for t in train_Y]
        test_Y = [mapping[t.split("/")[0]] for t in test_Y]
        assert len(set(train_Y).difference(set(mapping.values()))) == 0, 'Error in mapping train tagset'
        assert len(set(test_Y).difference(set(mapping.values()))) == 0, 'Error in mapping test tagset'

    # train the encoder
    le = preprocessing.LabelEncoder()
    le.fit(train_Y + test_Y)

    # transform tags
    train_Y = le.transform(train_Y)
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

def e_add(utt1,utt2):
    '''
    returns a function that adds two lists elementwise.

    :param utt1:
    :param utt2:
    :return:
    '''
    return map(add, utt1, utt2)

def represent_mix_simple(utterances, tags, model, op):
    '''
    incorporates previous utterance with operator param.
    retrieves utterance representations via inference
    :param labels:
    :param model:
    :return:
    '''

    # utterances: tag + utt.conversation_no + utt.caller + c
    # this represents every utterance as it's own embedding this should be extended to encorporate some context
    X = []
    for j, utt in enumerate(utterances):
        actual_utt = list(model.infer_vector(utt.split()))
        if j == 0:
            # if its the first utt in the list, we dont have a prev utterance. Repeat the actual utt.
            mixed_rep = op(actual_utt,actual_utt)
        else:
            # if its the beginning of a convo, we dont have a prev utterance. Repeat the actual utt.
            prev_conv_nr = tags[j-1].split('/')[1].split('_')[0]
            actual_conv_nr = tags[j].split('/')[1].split('_')[0]
            prev_utt = list(model.infer_vector(utterances[j-1])) if prev_conv_nr == actual_conv_nr else actual_utt
            mixed_rep = op(prev_utt,actual_utt)
        X.append(mixed_rep)
        if (j+1) % 1000 == 0:
            sys.stderr.write("\r");
            sys.stderr.write("utterance %s" % (j+1));
            sys.stderr.flush();

    sys.stderr.write("\r");
    sys.stderr.write("utterance %s" % (j+1));
    sys.stderr.flush();
    sys.stderr.write("\n")

    return X

def get_lookedup_vector(model, key):
    '''
    doc2vec returns an ndim-3 array if it doesnt find the key,
    :param model:
    :param key:
    :return:
    '''
    vec = model.docvecs[key]

    # if the doc2vec model returns trash, return None
    if not vec.ndim == 1:
        raise Doc2VecKeyError

    return vec

def represent_mix_lookup(labels, model, op):
    '''
    incorporates previous utterance with operator param.
    retrieves utterance representations via lookup
    :param labels:
    :param model:
    :return:
    '''

    i = 0   # nr of keys not found
    X = []
    for j, label in enumerate(labels):

        try:
            actual_vector = get_lookedup_vector(model, label)
        except Doc2VecKeyError:
            i += 1
            continue

        actual_utt = list(actual_vector)

        if j == 0:
            mixed_rep = op(actual_utt, actual_utt)
        else:
            prev_conv_utt = labels[j-1].split('/')[1].split('_')[0]
            actual_conv_utt = label.split('/')[1].split('_')[0]

            try:
                prev_vector = get_lookedup_vector(model, labels[j-1])
            except Doc2VecKeyError:
                i += 1
                continue

            prev_utt = list(prev_vector) if prev_conv_utt == actual_conv_utt else actual_utt
            mixed_rep = op(prev_utt, actual_utt)

        X.append(mixed_rep)
        if (j+1) % 1000 == 0:
            sys.stderr.write("\r");
            sys.stderr.write("utterance %s" % (j+1));
            sys.stderr.flush();

    sys.stderr.write("\r");
    sys.stderr.write("utterance %s" % (j+1));
    sys.stderr.flush();
    sys.stderr.write("\n")

    if i > 0:
        sys.stderr.write('## WARNING: Some samples were discarded!\n')

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

    train_Y, test_Y = encode_tags(train_Y, test_Y)

    print "BOW representation created"
    print "KNN Accuracy: ", cl.KNN_classifier(train_X, train_Y, test_X, test_Y)
    # print "SVM Accuracy: ", cl.SVM_classifier(train_X, train_Y, test_X, test_Y)
    print "NB Accuracy: ", cl.NB_classifier(train_X, train_Y, test_X, test_Y)


def evaluate_model(embedding_model_location, with_context=False, f=e_add, aggregated_tagset=False):
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

    if with_context:
        # ---------- lqrz: add or concatenate the previous utterance
        # f = concat
        # f = e_add
        test_X = represent_mix_simple(test_utt, test_Y, embedding_model, f)
        train_X = represent_mix_lookup(train_Y, embedding_model, f)
        # ----------
    else:
        train_X = represent_lookup(train_Y, embedding_model)
        test_X = represent_simple(test_utt, embedding_model)


    # encode tags
    train_Y, test_Y = encode_tags(train_Y, test_Y, aggregated_tagset=aggregated_tagset)

    # print np.array(train_X).shape, np.array(test_X).shape, np.array(train_Y).shape, np.array(test_Y).shape
    print "Training classifiers"
    # Train classifiers, print scores
    print "Model: ", embedding_model_location
    print "KNN Accuracy: ", cl.KNN_classifier(train_X, train_Y, test_X, test_Y)
    # print "SVM Accuracy: ", cl.SVM_classifier(train_X, train_Y, test_X, test_Y)
    print "NB Accuracy: ", cl.NB_classifier(train_X, train_Y, test_X, test_Y)
    print "MLP Accuracy: ", cl.MLP_classifier(train_X, train_Y, test_X, test_Y, n_iter=10)


def classify_context_dependent():
    # load data as:
    # training:
        # sequences of DA tags with corresponding speaker sequences for discourse model
        # utterances with corresponding DA tags for language model
    # testing:
        # sequences of utterances with corresponding speakers and DA tags (for evaluation)
    embedding_model_location = 'data/swda_bnc_noint_50_300'
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
    emmision_probability_matrix = np.vstack((emmision_probabilites,)*len(dialog)).T # for testing

    print "Loading sentence model"
    # train/load 'language model'
    embedding_model = Doc2Vec.load(embedding_model_location+'.model')
    # load a doc2vec model to train the language model with
    # vectorizer = CountVectorizer(min_df=1)
    # fit + transform on all training data
    # train_utt =[]
    train_Y = []
    train_X = []
    for dialog in train_dialogs:
        for tag, utt in dialog:
            # train_utt.append(utt)
            train_X.append(embedding_model.infer_vector(utt))
            train_Y.append(tag.split('/')[0])
    # train_X = embedding_model.infer_vector(train_utt)   
    model = mlp.Classifier(layers=[mlp.Layer("Sigmoid", units=100), mlp.Layer("Softmax")], learning_rate=0.001, n_iter=25)
    model.fit(np.array(train_X), np.array(train_Y))




    # learn transition matrix
    speaker_sequences = []
    tag_sequences = []
    for dialog in train_dialogs:
        speaker_sequences.append([tag[0].split('/')[1].split('_')[1] for tag in dialog])
        tag_sequences.append(le.transform([tag[0].split('/')[0] for tag in dialog]))
    print "Learning a Transition Matrix"
    start_prob, transition_matrix = hmm.learn_transition_matrix(tag_sequences, speaker_sequences, number_of_states = 43, order = 1)


    print "Decoding"
    # evauluation:
    correct = 0
    total = 0
    probas = []
    i = 0
    for dialog in test_dialogs:
        print i
        i += 1
        for tag, utt in dialog:
            rep = embedding_model.infer_vector([utt]) # TODO This does something very weird
            probas.append(model.predict_proba(rep))
        # this needs to be given the representation it needs in the language model
        # then we can use this with the predict_proba to calculate the emmision probability matrix

        # emmision_probability_matrix = np.vstack((emmision_probabilites,)*len(dialog)).T # for testing
        # print emmision_probability_matrix
        emmision_probability_matrix = np.vstack(probas).T
        # print emmision_probability_matrix

        true_tags = le.transform([tag[0].split('/')[0] for tag in dialog])
        speakers = [tag[0].split('/')[1].split('_')[1] for tag in dialog]
        predicted = hmm.viterbi_decoder(speakers,speakers, start_prob, transition_matrix,  emmision_probability_matrix, order = 1)
        score = hmm.evaluate_sequence(predicted, true_tags)
        correct += score
        total += len(dialog)
    print float(correct)/total
        # for every testing dialog, decode with viterbi
    # calculate accuracy maybe confusion matrix


if __name__ == '__main__':
    # classify_context_dependent()
    # train_utt, train_Y, test_utt, test_Y = load_data()
    # baseline_scores(train_utt, train_Y, test_utt, test_Y)

    # if with_context selected, then choose an agg function such as 'concat' or 'e_add'
    evaluate_model('data/models/swda_only_10_300', with_context=True, f=e_add, aggregated_tagset=True)