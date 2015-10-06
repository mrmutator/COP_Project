__author__ = 'rwechsler'
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from corpora import get_swda_utterances, get_SB_utterances
import random
import pickle

def get_similar_utterances(utt, model, utterances):
    utt_vector = model.infer_vector(utt.split())

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



if __name__ == "__main__":

    utterance_tag_set = []
    utterances = dict()
    for i, (tag, utt_tokens) in enumerate(get_swda_utterances("data/swda")):
        utterance_tag_set.append(TaggedDocument(utt_tokens, [unicode(tag + '_%s' % i)]))
        utterances[unicode(tag + "_%s"%i)] = " ".join(utt_tokens)


    print len(utterances)

    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=4)
    model.build_vocab(utterance_tag_set)

    # use pretrained word2vec model to initialize word parameters
    model.intersect_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz", binary=True)

    for epoch in range(10):
        print epoch
        random.shuffle(utterance_tag_set)
        model.train(utterance_tag_set)


    save_all_models(model, utterances, "data/test")

    #model.save("test.model")

    #model = Doc2Vec.load("test.model")

    get_similar_utterances("thank you", model, utterances)

    get_similar_utterances("I live in Amsterdam .", model, utterances)












