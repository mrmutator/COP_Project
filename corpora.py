__author__ = 'rwechsler'
from swda import CorpusReader
import glob
from nltk import word_tokenize
import re
import codecs
import os
from lxml import etree as ET
import random


def get_swda_utterances(swda_dir):
    corpus = CorpusReader(swda_dir)
    c = 0
    for trans in  corpus.iter_transcripts(display_progress=True):
            for utt in trans.utterances:
                utt_temp = re.sub(r'\(|\)|-|\{.+? |\}|\[|\]|\+|#|/|<.+?>', "", utt.text.lower())
                utt_tokens = word_tokenize(re.sub("<|>", "", utt_temp))
                if utt_tokens:
                    yield utt.damsl_act_tag() + "/%s_%s_%s" % (utt.conversation_no, utt.caller, c), utt_tokens
                    c += 1

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

def write_file(corpus, outfile_name):
    outfile = codecs.open(outfile_name, "w", "utf-8")
    for tag, utt_tokens in corpus:
        outfile.write(tag + "\t" + " ".join(utt_tokens) + "\n")
    outfile.close()

def get_utterances_from_file(file_name):
    infile = codecs.open(file_name, "r", "utf-8")
    for line in infile:
        tag, utt = line.strip().split("\t")
        yield tag, utt.split()

def get_bnc_utterances(bnc_dir):
    for root, dirs, files in os.walk(bnc_dir):
        for name in files:
            file_name =  os.path.join(root, name)
            if file_name.endswith(".xml"):
                print "parsing " + file_name
                tree = ET.parse(open(file_name, "r"))
                doc_id = tree.getroot().attrib['{http://www.w3.org/XML/1998/namespace}id'].strip()
                for turn in tree.xpath("//u"):
                    for utt in turn.xpath("s"): # utterances in BNC mean turns, so we use one sentence s as utterance
                        utt_id =  utt.attrib['n']
                        tokens = [w.text.lower().strip() for w in utt.xpath(".//w|c")]
                        if tokens:
                            yield doc_id + "_" + utt_id, tokens



def write_train_test_files(corpus, transcript_number_file, number_of_test_transcripts, test_file_name, train_file_name):
    transcript_number_file = open(transcript_number_file, "r")
    transcript_numbers = transcript_number_file.readlines()
    transcript_number_file.close()
    transcript_numbers = map(int, (map(str.strip, transcript_numbers)))
    test_set_transcripts = random.sample(transcript_numbers, number_of_test_transcripts)

    outfile_test = codecs.open(test_file_name, "w", "utf-8")
    outfile_train = codecs.open(train_file_name, "w", "utf-8")
    test_tags = set()
    for tag, utt_tokens in corpus:
        if int(tag.split("/")[-1].split("_")[0]) in test_set_transcripts:
            outfile_test.write(tag + "\t" + " ".join(utt_tokens) + "\n")
            test_tags.add(tag.split("/")[0])
        else:
            outfile_train.write(tag + "\t" + " ".join(utt_tokens) + "\n")

    outfile_train.close()
    outfile_test.close()
    print "tags in test set: ", len(test_tags)



if __name__ == "__main__":
    # write preprocessed utterances files to speed up further processing
    corpus = get_swda_utterances("data/swda")
    # corpus = get_utterances_from_file("data/swda_utterances.txt")
    #write_file(corpus, "data/swda_utterances.txt")
    write_train_test_files(corpus, "data/transcripts.txt", 19, "test", "train")

    # corpus = get_SB_utterances("data/SB")
    # write_file(corpus, "data/SB_utterances.txt")

    # corpus = get_bnc_utterances("data/BNC_XML/Texts")
    # write_file(corpus, "data/BNC_utterances.txt")

