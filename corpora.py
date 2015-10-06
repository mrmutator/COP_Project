__author__ = 'rwechsler'
from swda import CorpusReader
import glob
from nltk import word_tokenize
import re
import codecs

def get_swda_utterances(swda_dir):
    corpus = CorpusReader(swda_dir)

    for trans in  corpus.iter_transcripts(display_progress=True):
            for utt in trans.utterances:
                utt_tokens =  word_tokenize(re.sub(r'\{.+? |\}|\[|\]|\+|#|/|<.+?>', "", utt.text.lower()))
                if utt_tokens:
                    yield utt.damsl_act_tag(), utt_tokens

def get_SB_utterances(SB_dir):
    for f in glob.glob(SB_dir + "/*.trn"):
        line_no = 0
        infile = codecs.open(f, "r", "latin-1")
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

def write_swda_file(swda_dir, outfile_name):
    outfile = codecs.open(outfile_name, "w", "utf-8")
    for tag, utt_tokens in get_swda_utterances(swda_dir):
        outfile.write(tag + "\t" + " ".join(utt_tokens) + "\n")
    outfile.close()

def get_swda_utterances_from_file(swda_file):
    infile = codecs.open(swda_file, "r", "utf-8")
    for line in infile:
        try:
            tag, utt = line.strip().split("\t")
        except ValueError:
            print "---->>>>>>", line.strip(), line.split("\t")
            break
        yield tag, utt.split()



if __name__ == "__main__":
    # for tag, utt in get_SB_utterances("data/SB"):
        # print tag, utt
    write_swda_file("data/swda", "data/swda_file.txt")

    for tag, tokens in get_swda_utterances_from_file("data/swda_file.txt"):
        print tag, tokens
