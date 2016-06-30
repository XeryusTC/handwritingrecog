import sys, os, inspect, shutil, fnmatch, logging
import numpy as np
import general.preprocessor as prep
import create_segments
import general.hog as hog
from recognize.label_words import Recognizer
import pickle
import create_lexicon
import create_probTables

import xml.etree.ElementTree as ET
from unipath import Path, DIRS_NO_LINKS
import cv2

# Set to True if run for the first time
create_segments = False
create_lexicon = False
create_tables = False

def main():
    # Directories
    sentenceDir = Path('tmp/sentences/')
    wordDir = Path('tmp/words/')

    # Logging
    logging.basicConfig(format='%(asctime)s %(levelname)-8s: %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')

    # Check commandline parameters
    if len(sys.argv) != 4:
        print "Usage: python %s image.ppm input.words out.words" % sys.argv[0]
        sys.exit(1)

    img = cv2.imread(sys.argv[1], 0)
    words_file_name = sys.argv[2]

    # Find out the used dataset
    dataset = None
    for dset in ['Stanford', 'KNMP']:
        if fnmatch.fnmatch(os.path.basename(words_file_name), dset+'*.words'):
            dataset = dset

    if dataset == None:
        print "Usage: python %s image.ppm input.words /path/to/output.words" % sys.argv[0]
        print "\tDataset should be either 'KNMP' or 'Stanford'"
        sys.exit(1)

    if create_segments:
        logging.info("Creating segments for dataset %s", dataset)
        create_segments.create(dataset)
        logging.info("Segments created for dataset %s", dataset)

    # Preprocess
    img = prep.preprocess(img)

    # Get the lexicon
    lexicon = {}

    # Get the sorted unique list of class labels
    featureDir = 'tmp/features/'
    trainDir = featureDir + 'train/'
    trainLabels = np.load(trainDir + 'labels.npy')
    classes = sorted(set(trainLabels))

    if create_lexicon:
        lexicon = create_lexicon.create_lexicon()
    else:
        with open("tmp/lexicon.csv") as f:
            for line in f:
                (key, val) = line.split(',')
                lexicon[key] = int(val)

    # Get probabiliity tables
    if create_tables:
        stateProbs = create_probTables.create_stateProbs(lexicon)
        transProbs = create_probTables.create_transProbs(lexicon)
    else:
        stateProbs = pickle.load(open("stateProbs.pickle"))
        transProbs = pickle.load(open("transProbs.pickle"))


    # Recognize the words
    xml = ET.parse(words_file_name).getroot()
    recog = Recognizer(sentenceDir, wordDir, xml, img)
    for word, word_img in recog.next_word():
        logging.info("Word: %s" % word.get('text'))
        cuts = recog.find_cuts(word_img)
        if cuts is not None:
            cuts.insert(0, 0) # Also create a window at the start of the word
            estimate = recog.recursiveRecognize(word_img, cuts, lexicon, stateProbs, transProbs, classes)
            logging.info("estimate: %s" % estimate)
            # text, candidates = recog.recognize(word_img, cuts, lexicon, stateProbs, transProbs)
            # correctText = word.get('text')
            # print "Word in candidates: ", correctText in candidates
            # print "Correct text: ", correctText
            # print "Estimated word: ", text, "\n"
            # word.set('text', text)
        else:
            continue
    ET.ElementTree(recog.words).write(sys.argv[3])

if __name__ == '__main__':
    main()
