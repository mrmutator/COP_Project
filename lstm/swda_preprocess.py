"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='../data/swda_utterances'
model_path = '../data/test'

import numpy
import cPickle as pkl
from collections import OrderedDict
import glob
import os,sys
sys.path.append('/Users/davwo/Documents/Computational Pragmatics/Project/COP_Project')
from subprocess import Popen, PIPE
from nltk import word_tokenize
from par2vec import load_all_models
from gensim.models import Doc2Vec
from corpora import get_utterances_from_file
from collections import defaultdict
import numpy as np



def grab_data(path, model):
    dialogs = defaultdict(list)
    speaker_dict = {"A": 0, "B" : 1}
    for tag, utterance in get_utterances_from_file(path):
        transcript_speaker_id = tag.split('/')[1].split("_")
        # we might want to look up the vector in stead of inferring it?
        utterance_representation = np.append(model.infer_vector(utterance), speaker_dict[transcript_speaker_id[1]])
        dialogs[transcript_speaker_id[0]].append(utterance_representation)
    
    # we are using dummy (random) tags, as we train just to get embeddings (is this ok?) maybe we nee to re assign every time?
    return dialogs.values(), np.random.randint(2, size = len(dialogs.values()))

def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    model, utt_train = load_all_models(model_path)
    path = dataset_path

    train_x, train_y= grab_data(path+'.train', model)
    test_x, test_y = grab_data(path+'.test', model)
    
    f = open('swda_embedded_'+ model_path.split('/')[-1] + '.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()


if __name__ == '__main__':
    main()