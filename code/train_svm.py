from __future__ import print_function
import logging
import pickle
import sys
from unipath import Path

from general.hog import doHog
from general.create_sets import create_sets
from train.svm import runSVM
import create_segments

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] != "--create_segments":
        print("Usage: python %s [--create_segments]" % sys.argv[0])

    need_create_segments = len(sys.argv) == 2

    segmentDir = Path('tmp/segments/')
    featureDir = Path('tmp/features/')
    trainDir = featureDir + 'train/'
    testDir  = featureDir + 'test/'

    if need_create_segments:
        logging.info("Create segments of the images")
        create_segments.create_seg()

    logging.info("Creating HOG features")
    doHog(segmentDir, featureDir)

    logging.info("Creating train and test set")
    create_sets(featureDir, 0.7)

    logging.info("Running SVM")
    SVM, accuracy = runSVM(trainDir, testDir)
    logging.info("SVM accuracy: %f" % accuracy)
    with open('tmp/svm.pickle', 'w') as f:
        pickle.dump(SVM, f)
    logging.info("Done")
