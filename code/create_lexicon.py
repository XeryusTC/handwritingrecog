# Packages
import sys, os, cv2, uuid, logging, csv
import logging.config
import numpy as np
from unipath import Path, DIRS_NO_LINKS
from scipy.cluster.vq import whiten, kmeans2
import xml.etree.ElementTree as ET

# Own modules
from general.preprocessor import preprocess

logging.config.fileConfig('logging.conf')

def main():
    create()

def create_lexicon():
    lex = {}
    with open("tmp/lexicon.txt") as f:
        for line in f:
            (key, val) = line.split()
            lex[key] = int(val)
    # Add our own lexicon
    lex = combine_lexicons(lex)
    return lex

def create_own_lexicon():

    lexicon = {}

    for dataset in ['KNMP', 'Stanford']:
        # Find all the annotated pages in the dataset
        ann_dir = Path(Path.cwd().ancestor(1), 'data/hwr_data/words/' + dataset)
        annotations = ann_dir.listdir( '*.words')

        for f in annotations:
            # Segment
            annotation = ET.parse(f).getroot()
            for word in annotation.iter('Word'):
                text = word.get('text')

                # Add word to lexicon
                if lexicon.has_key(text):
                    lexicon[text] += 1
                else :
                    lexicon[text] = 1
    return lexicon

def combine_lexicons(orig_lex):
    own_lex = create_own_lexicon()

    for word, number in own_lex.iteritems():
        if orig_lex.has_key(word):
            orig_lex[word] = max(orig_lex[word], own_lex[word])
        else:
            orig_lex[word] = number
    return orig_lex

def create():
    # create a clean temporary directory
    work_dir = Path("tmp")
    work_dir.mkdir()

    lexicon = create_lexicon()

    lexicon_path = Path(work_dir, "lexicon.csv")
    w = csv.writer(open(lexicon_path, "w"))
    for key, val in lexicon.items():
        w.writerow([key, val])

if __name__ == '__main__':
    main()
