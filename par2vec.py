__author__ = 'rwechsler'
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy
from swda import CorpusReader
from nltk import word_tokenize
import re


def get_utterances(swda_dir):
    corpus = CorpusReader('swda')

    for trans in  corpus.iter_transcripts(display_progress=True):
            for utt in trans.utterances:
                yield utt.damsl_act_tag(), utt.text



utterances = []
for i, (tag, utt) in enumerate(get_utterances("swda")):
    utt_tokens =  word_tokenize(re.sub(r'\{.+? |\}|\[|\]|\+|#|/|<.+?>', "", utt))
    utterances.append(TaggedDocument(utt_tokens, [unicode(tag + '_%s' % i)]))


model = Doc2Vec(documents=utterances,min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
model.save("test.model")

model = Doc2Vec.load("test.model")










