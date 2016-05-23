# -*- coding: utf-8 -*-
import numpy as np
import logging
import logging.config
from unipath import Path, DIRS_NO_LINKS
import cv2
import time

from general import hog
from train.svm import runSVM
from general.create_sets import create_sets

IMG_DIR = Path('tmp/segments/')
HOG_DIR = Path('tmp/features/')
FEATURE_DIR = Path('tmp/features/')

def do_hog(hog_type='xeryus', char_size=(72, 72), window_size=(80, 80),
        block_size=(2, 2), cell_size=(8, 8), nbins=9):
    # reset hog tree
    features = []
    labels = []
    testFeatures = []
    testLabels = []

    FEATURE_DIR.rmtree()
    FEATURE_DIR.mkdir()

    logging.debug("Creating test and training set")
    for label in IMG_DIR.listdir(filter=DIRS_NO_LINKS):
        for f in label.listdir(pattern='*.ppm'):
            img = cv2.imread(f)
            if hog_type == 'xeryus':
                hist = hog.hog_xeryus(img, char_size, window_size, block_size,
                    cell_size, nbins)
            else:
                hist = hog_alternative(img)
            features.append(hist[:,0])
            labels.append(str(label.name))
    logging.info("HOG feature size: %d", len(hist))

    np.save(FEATURE_DIR + 'hog', features)
    np.save(FEATURE_DIR + 'labels', labels)

if __name__ == '__main__':
    logging.config.fileConfig('logging.conf')
    assert IMG_DIR.exists(), "You must first create segmented images"

    block_sizes = ((1, 1), (2, 2), (3, 3), (4, 4))
    cell_sizes  = ((4, 4), (6, 6), (8, 8), (12, 12), (16, 16))

    results = {}

    start_time = time.time()
    for cell_size in cell_sizes:
        for block_size in block_sizes:
            svm_time = time.time()

            logging.info("Block size: %s Cell size: %s", block_size, cell_size)
            logging.debug("Creating HOG features")
            do_hog(block_size=block_size, cell_size=cell_size,
                    char_size=(88, 88), window_size=(96, 96))

            logging.debug("Creating test and training sets")
            create_sets(FEATURE_DIR)

            logging.debug("Training SVM")
            SVM, accuracy = runSVM(FEATURE_DIR + 'train/',
                FEATURE_DIR + 'test/')
            logging.info("SVM training time: %f s", time.time() - svm_time)
            logging.info("SVM accuracy: %f", accuracy)

    logging.info("Total execution time: %f s", time.time() - start_time)
