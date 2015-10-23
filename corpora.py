__author__ = 'rwechsler'
from swda import CorpusReader
import glob
from nltk import word_tokenize
import re
import codecs
import os
from lxml import etree as ET


def get_swda_utterances(swda_dir):
    corpus = CorpusReader(swda_dir)
    c = 0
    last_utterances = dict()

    for trans in  corpus.iter_transcripts(display_progress=True):
            last_utterances["A"] = []
            last_utterances["B"] = []
            for utt in trans.utterances:
                utt_temp = re.sub(r'\(|\)|-|\{.+? |\}|\[|\]|\+|#|/|<.+?>|,', "", utt.text.lower())
                utt_tokens = word_tokenize(re.sub("<|>", "", utt_temp))
                if utt.damsl_act_tag() != "+":
                    last_utterances[utt.caller].append((c, utt.damsl_act_tag() + "/%s_%s_%s" % (utt.conversation_no, utt.caller, c), utt_tokens))
                    c += 1
                else:
                    try:
                        prev = last_utterances[utt.caller].pop()
                        new = (prev[0], prev[1], prev[2] + utt_tokens)
                        last_utterances[utt.caller].append(new)
                    except IndexError:
                        pass
                        # RW: for some reason, Chris Potts' Corpus Reader gives us utterances with a "+" tag although
                        # there is no previous utterance of the same speaker to complete.
                        # Looking at the originial data, there seems to be a bug in his Corpus Reader that skips some
                        # stuff in the beginning for some reason (e.g. the beginning of conv. no 3554.
                        print utt.conversation_no
            utterances = last_utterances["A"] + last_utterances["B"]
            utterances = sorted(utterances, key= lambda t: t[0])
            for tpl in utterances:
                if tpl[2]:
                    yield tpl[1:]

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

def get_transcript_numbers(file_name):
    transcript_number_file = open(file_name, "r")
    transcript_numbers = transcript_number_file.readlines()
    transcript_number_file.close()
    return [int(x[2:].strip()) for x in transcript_numbers]


def write_train_test_files(corpus, train_transcript_number_file, test_transcript_number_file, test_file_name, train_file_name):
    train_numbers = get_transcript_numbers(train_transcript_number_file)
    test_numbers = get_transcript_numbers(test_transcript_number_file)

    outfile_test = codecs.open(test_file_name, "w", "utf-8")
    outfile_train = codecs.open(train_file_name, "w", "utf-8")
    test_tags = set()
    for tag, utt_tokens in corpus:
        conv_number = int(tag.split("/")[-1].split("_")[0])
        if conv_number in test_numbers:
            outfile_test.write(tag + "\t" + " ".join(utt_tokens) + "\n")
            test_tags.add(tag.split("/")[0])
        elif conv_number in train_numbers:
            outfile_train.write(tag + "\t" + " ".join(utt_tokens) + "\n")
        else:
            # print conv_number # should not happen, happens nevertheless?!
            pass

    outfile_train.close()
    outfile_test.close()
    print "tags in test set: ", len(test_tags)



if __name__ == "__main__":
    # write preprocessed utterances files to speed up further processing
    corpus = get_swda_utterances("data/swda")
    # corpus = get_utterances_from_file("data/swda_utterances.txt")
    #write_file(corpus, "data/swda_utterances.txt")
    write_train_test_files(corpus, "data/ws97-train-convs.list", "data/ws97-test-convs.list", "test", "train")

    # corpus = get_SB_utterances("data/SB")
    # write_file(corpus, "data/SB_utterances.txt")

    # corpus = get_bnc_utterances("data/BNC_XML/Texts")
    # write_file(corpus, "data/BNC_utterances.txt")

