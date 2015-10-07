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
                print "parsing" + file_name
                tree = ET.parse(open(file_name, "r"))
                doc_id = tree.getroot().attrib['{http://www.w3.org/XML/1998/namespace}id']
                for turn in tree.xpath("//u"):
                    for utt in turn.xpath("s"): # utterances in BNC mean turns, so we use one sentence s as utterance
                        utt_id =  utt.attrib['n']
                        tokens = [w.text.lower() for w in utt.xpath(".//w|c")]
                        yield doc_id + "_" + utt_id, tokens





if __name__ == "__main__":
    # for tag, utt in get_SB_utterances("data/SB"):
        # print tag, utt
    # corpus = get_swda_utterances("data/swda_file.txt")
    # write_file(corpus, "data/swda_file.txt")
    #
    #for tag, tokens in get_swda_utterances_from_file("data/swda_file.txt"):
    #     print tag, tokens
    #corpus = get_bnc_utterances("data/BNC_XML/Texts/")
    #write_file(corpus, "BNC_utterances.txt")
    pass

