__author__ = 'rwechsler'
from swda import CorpusReader
import glob
from nltk import word_tokenize
import re

def get_swda_utterances(swda_dir):
    corpus = CorpusReader(swda_dir)

    for trans in  corpus.iter_transcripts(display_progress=True):
            for utt in trans.utterances:
                utt_tokens =  word_tokenize(re.sub(r'\{.+? |\}|\[|\]|\+|#|/|<.+?>', "", utt.text.lower()))
                yield utt.damsl_act_tag(), utt_tokens

def get_SB_utterances(SB_dir):
    for f in glob.glob(SB_dir + "/*.trn"):
        line_no = 0
        infile = open(f, "r")
        for line in infile:
            line_no += 1
            elements = line.strip().split("\t")
            if len(elements) == 4:
                _, _, name, utt = elements
            elif len(elements) == 3:
                _, name, utt = elements
            else:
                # idiots!
                continue
            if name.startswith(">"):
                 continue # no utterance
            temp_utt = re.sub("<[A-Z]+|<VOX|[A-Z]+>|VOX>", "", utt)
            utt_tokens = word_tokenize(re.sub(r"\(.+?\)|\.\.+|\[\d?|\d?\]|-|@|=|<|>|#|%|\+|\w+_|X+", "", temp_utt).lower())
            if utt_tokens:
                yield f.split("/")[-1] + "_ %s" % line_no, utt_tokens

        infile.close()


if __name__ == "__main__":
    for tag, utt in get_SB_utterances("data/SB"):
        print tag, utt