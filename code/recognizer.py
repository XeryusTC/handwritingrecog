import sys, os, inspect, shutil, fnmatch, logging
import numpy as np
import general.preprocessor as prep
import create_segments
import general.hog as hog
from recognize.label_words import Recognizer
import pickle
import create_lexicon
import create_probTables
import create_lexicon_means_stds

import xml.etree.ElementTree as ET
from unipath import Path, DIRS_NO_LINKS
import cv2

# Set to True if run for the first time
create_lexicon_stuff = True

def reduce_lexicon(cuts, word_img, lexicon, lexicon_means_stds):
    reduced_lexicon = {}
    # The number of cuts is the max number of letters
    for word, number in lexicon.iteritems():
        if len(word) <= len(cuts):

            if (
                lexicon_means_stds[word][0] - 1 * lexicon_means_stds[word][1] < word_img.shape[1] and
                lexicon_means_stds[word][0] + 1 * lexicon_means_stds[word][1] > word_img.shape[1]
            ):
                reduced_lexicon[word] = number

    reduction =  (1-float(len(reduced_lexicon))/len(lexicon) )*100
    logging.info("\tReduced lexicon by: %s percent" % reduction )

    return reduced_lexicon, reduction

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

    # Read input files
    img = cv2.imread(sys.argv[1], 0)
    words_file_name = sys.argv[2]

    # Preprocess image
    img = prep.preprocess(img)

    # Get the sorted unique list of class labels
    featureDir = 'tmp/features/'
    trainDir = featureDir + 'train/'
    trainLabels = np.load(trainDir + 'labels.npy')
    classes = sorted(set(trainLabels))

    # Get lexicon information
    if create_lexicon_stuff:
        lexicon = create_lexicon.create_lexicon()
        lexicon_means_stds = create_lexicon_means_stds.create()
        stateProbs = create_probTables.create_stateProbs(lexicon)
        transProbs = create_probTables.create_transProbs(lexicon)
    else:
        with open("tmp/lexicon.csv") as f:
            for line in f:
                (key, val) = line.split(',')
                lexicon[key] = int(val)
        lexicon_means_stds = pickle.load(open("tmp/lexicon_means_stds.pickle"))
        stateProbs = pickle.load(open("tmp/stateProbs.pickle"))
        transProbs = pickle.load(open("tmp/transProbs.pickle"))

    # print lexicon

    # Recognition accuracy names
    correct = 0
    false = 0
    inLex = 0
    avgReduction = 0

    # Recognize the words
    xml = ET.parse(words_file_name).getroot()
    recog = Recognizer(sentenceDir, wordDir, xml, img)
    for word, word_img in recog.next_word():
        required_word = word.get('text')
        logging.info("Word: %s" % required_word)
        cuts = recog.find_cuts(word_img)
        if cuts is not None:
            reduced_lexicon, reduction = reduce_lexicon(cuts, word_img, lexicon, lexicon_means_stds)
            avgReduction += reduction

            cuts.insert(0, 0) # Also create a window at the start of the word
            estimate = recog.recursiveRecognize(word_img, cuts, reduced_lexicon, stateProbs, transProbs, classes)
            logging.info("Estimate: %s" % estimate)

            if required_word in reduced_lexicon:
                logging.info("\tIs the word in reduced lexicon: yes")
                inLex += 1
            else:
                logging.info("\tIs the word in reduced lexicon: no")
            if required_word == estimate:
                correct += 1
            else:
                false += 1
            print('\n')
        else:
            continue
    ET.ElementTree(recog.words).write(sys.argv[3])
    accuracy = float(correct)/(correct+false) * 100
    totalInLex = float(inLex)/(correct+false) * 100
    avgReduction = avgReduction/float(correct+false)
    logging.info("Correct: %s, False: %s\n \tAccuracy: %s" % (correct, false, accuracy) )
    logging.info("In lexicon: %s" % totalInLex)
    logging.info("Average reduction of lexicon: %s" % avgReduction)

if __name__ == '__main__':
    main()
