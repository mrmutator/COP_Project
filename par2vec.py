__author__ = 'rwechsler'
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from corpora import get_utterances_from_file
import random
import pickle

def get_similar_utterances(utt, model, utterances):
    utt_vector = model.infer_vector(utt.lower().split())

    similar_utterances = model.docvecs.most_similar([utt_vector])

    print "similar utterances to '" + utt + "':"
    for utt_label, sim in similar_utterances:
        print utterances[utt_label], sim

def save_all_models(model, utterances, file_name):
    model.save(file_name + ".model")
    pickle.dump(utterances, open(file_name + ".utt", "wb"))

def load_all_models(file_name):
    model = Doc2Vec.load(file_name + ".model")
    utterances = pickle.load(open(file_name + ".utt", "rb"))
    return model, utterances


def train_model(train_corpus_file, output_file_name, min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=4, alpha=0.025,
                min_alpha = 0.001, epochs=10, w2v_intersect=None):
    utterance_tag_set = []
    utterances = dict()
    for tag, utt_tokens in get_utterances_from_file(train_corpus_file):
        utterance_tag_set.append(TaggedDocument(utt_tokens, [unicode(tag )]))
        utterances[unicode(tag)] = " ".join(utt_tokens)


    print "Training size:", len(utterances)

    model = Doc2Vec(min_count=min_count, window=window, size=size, sample=sample, negative=negative, workers=workers)
    model.build_vocab(utterance_tag_set)

    # use pretrained word2vec model to initialize word parameters
    if w2v_intersect:
        model.intersect_word2vec_format(w2v_intersect, binary=True)


    alpha_delta = (alpha - min_alpha) / epochs
    for epoch in range(epochs):
        print "Epoch: ", epoch
        random.shuffle(utterance_tag_set)
        model.alpha, model.min_alpha = alpha, alpha
        model.train(utterance_tag_set)
        alpha -= alpha_delta


    save_all_models(model, utterances, output_file_name)


def test_model(model_name):
    model, utterances = load_all_models(model_name)

    get_similar_utterances("thank you", model, utterances)

    get_similar_utterances("I live in Amsterdam .", model, utterances)


if __name__ == "__main__":

    # train different dimensions
    for data_label, data in [("swda_only", "data/swda_utterances.train"), ("swda_bnc", "data/swda_bnc_utterances.train")]:
        for dim in [100, 400, 500, 800, 1000]:
            print data_label, dim
            train_model(data, "models/" + data_label + "_10" + "_" + str(dim), epochs=10, size=dim)
















