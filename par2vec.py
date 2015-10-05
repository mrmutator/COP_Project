__author__ = 'rwechsler'
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from swda import CorpusReader
from nltk import word_tokenize
import re
import random

def get_utterances(swda_dir):
    corpus = CorpusReader(swda_dir)

    for trans in  corpus.iter_transcripts(display_progress=True):
            for utt in trans.utterances:
                yield utt.damsl_act_tag(), utt.text.lower()


def get_similar_utterances(utt, model, utterances):
    utt_vector = model.infer_vector(utt.split())

    similar_utterances = model.docvecs.most_similar([utt_vector])

    print "similar utterances to '" + utt + "':"
    for utt_label, sim in similar_utterances:
        print utterances[utt_label], sim

if __name__ == "__main__":

    utterance_tag_set = []
    utterances = dict()
    for i, (tag, utt) in enumerate(get_utterances("swda")):
        utt_tokens =  word_tokenize(re.sub(r'\{.+? |\}|\[|\]|\+|#|/|<.+?>', "", utt))
        utterance_tag_set.append(TaggedDocument(utt_tokens, [unicode(tag + '_%s' % i)]))
        utterances[unicode(tag + "_%s"%i)] = utt


    print len(utterances)

    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=4)
    model.build_vocab(utterance_tag_set)

    # use pretrained word2vec model to initialize word parameters
    model.intersect_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz", binary=True)

    for epoch in range(10):
        print epoch
        random.shuffle(utterance_tag_set)
        model.train(utterance_tag_set)


    model.save("test.model")

    #model = Doc2Vec.load("test.model")

    get_similar_utterances("thank you", model, utterances)

    get_similar_utterances("I live in Amsterdam .", model, utterances)












