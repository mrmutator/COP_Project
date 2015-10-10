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



if __name__ == "__main__":

    utterance_tag_set = []
    utterances = dict()
    for tag, utt_tokens in get_utterances_from_file("data/swda_utterances.txt"):
        utterance_tag_set.append(TaggedDocument(utt_tokens, [unicode(tag )]))
        utterances[unicode(tag)] = " ".join(utt_tokens)

    # for tag, utt_tokens in get_utterances_from_file("data/SB_utterances.txt"):
    #     utterance_tag_set.append(TaggedDocument(utt_tokens, [unicode(tag)]))
    #     utterances[unicode(tag)] = " ".join(utt_tokens)

    for tag, utt_tokens in get_utterances_from_file("data/BNC_utterances.txt"):
        utterance_tag_set.append(TaggedDocument(utt_tokens, [unicode(tag)]))
        utterances[unicode(tag)] = " ".join(utt_tokens)

    print len(utterances)

    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=4)
    model.build_vocab(utterance_tag_set)

    # use pretrained word2vec model to initialize word parameters
    model.intersect_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz", binary=True)


    alpha, min_alpha, passes = (0.025, 0.001, 10)
    alpha_delta = (alpha - min_alpha) / passes
    for epoch in range(passes):
        print "Epoch: ", epoch
        random.shuffle(utterance_tag_set)
        model.alpha, model.min_alpha = alpha, alpha
        model.train(utterance_tag_set)
        alpha -= alpha_delta


    save_all_models(model, utterances, "data/test")

    #model, utterances = load_all_models("data/test")

    get_similar_utterances("thank you", model, utterances)

    get_similar_utterances("I live in Amsterdam .", model, utterances)












