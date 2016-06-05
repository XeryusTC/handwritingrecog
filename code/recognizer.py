import sys, os, inspect, shutil, fnmatch, logging
import general.preprocessor as prep
import create_segments
import general.hog as hog
from recognize.label_words import Recognizer

import xml.etree.ElementTree as ET
from unipath import Path, DIRS_NO_LINKS
import cv2

# Debug booleans
create_segments = False

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

    if create_segments is True:
        logging.info("Creating segments for dataset %s", dataset)
        create_segments.create(dataset)
        logging.info("Segments created for dataset %s", dataset)

    # Preprocess
    img = prep.preprocess(img)

    # Recognize the words
    xml = ET.parse(words_file_name).getroot()
    recog = Recognizer(sentenceDir, wordDir, xml, img)
    for word, word_img in recog.next_word():
        cuts = recog.find_cuts(word_img)
        if cuts is not None:
            cuts.insert(0, 0) # Also create a window at the start of the word
            text = recog.recognize(word_img, cuts)
            print word.get('text'), text
            word.set('text', text)
        else:
            continue
    ET.ElementTree(recog.words).write(sys.argv[3])

if __name__ == '__main__':
    main()
