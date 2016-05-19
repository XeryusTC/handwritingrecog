import sys, os, inspect, shutil, fnmatch, logging
import general.preprocessor as prep
import create_segments
import recognize.recognize as recognize
import general.hog as hog

import modules.toolbox.pamImage as pamImage
import xml.etree.ElementTree as ET
from unipath import Path, DIRS_NO_LINKS
import cv2

# Debug booleans
create_segments = False

def main():
    # Logging
    logging.basicConfig(format='%(asctime)s %(levelname)-8s: %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')

    # Check commandline parameters
    if len(sys.argv) != 4:
        print "Usage: python %s image.ppm input.words /path/to/output.words" % sys.argv[0]
        sys.exit(1)

    img = cv2.imread(sys.argv[1], 0)
    words_file_name = sys.argv[2]
    words_file_name_out = sys.argv[3]

    # Find out the used dataset
    dataset = None
    for dset in ['Stanford', 'KNMP']:
        if fnmatch.fnmatch(words_file_name, dset+'*.words'):
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
    prepImage = prep.preprocess(img)
    cv2.imwrite('preprocessed.ppm', prepImage)
    preIm = pamImage.PamImage("preprocessed.ppm")

    # Do the hog
    hog.main_xeryus()

    # Recognize the words
    e = ET.parse(words_file_name).getroot()
    returnedTree = recognize.recognize(preIm, e)
    returnedTree.write(words_file_name_out)

if __name__ == '__main__':
    main()
