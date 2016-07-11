from __future__ import print_function
import logging
import logging.config
from unipath import Path, DIRS_NO_LINKS
import sys
import cv2
import numpy as np
import time

from general import hog
from train.svm import runSVM
from general.create_sets import create_sets

IMG_DIR = Path('tmp/segments/')
HOG_DIR = Path('tmp/features/')
RESULTS_FILE = Path('tmp/svm_stats.csv')

def do_hog(char_size=(32, 32), window_size=(80, 80), block_size=(2, 2),
        cell_size=(8, 8), nbins=9, train_split=0.7):
    # reset hog tree
    features = []
    labels = []
    test_features = []
    test_labels = []

    HOG_DIR.rmtree()
    HOG_DIR.mkdir()
    train_dir = HOG_DIR + 'train/'
    test_dir = HOG_DIR + 'test/'
    train_dir.mkdir()
    test_dir.mkdir()

    logging.debug("Creating test and training set")
    for label in IMG_DIR.listdir(filter=DIRS_NO_LINKS):
        for f in label.listdir(pattern='*.ppm'):
            img = cv2.imread(f)
            hist = hog.hog_xeryus(img, char_size, window_size, block_size,
                    cell_size, nbins)
            # Split into training and test set
            if np.random.random_sample() <= train_split:
                features.append(hist[:,0])
                labels.append(str(label.name))
            else:
                test_features.append(hist[:,0])
                test_labels.append(str(label.name))
    logging.info("HOG feature size: %d", len(hist))

    np.save(HOG_DIR + 'train/hog.npy', features)
    np.save(HOG_DIR + 'train/labels.npy', labels)
    np.save(HOG_DIR + 'test/hog.npy', test_features)
    np.save(HOG_DIR + 'test/labels.npy', test_labels)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python %s <width> <height>" % sys.argv[0])
        sys.exit(1)
    assert IMG_DIR.exists(), "You must first create segmented images"
    logging.config.fileConfig('logging.conf')

    target_width = int(sys.argv[1])
    target_height = int(sys.argv[2])

    block_sizes = ((1, 1), (2, 2), (3, 3), (4, 4))
    cell_sizes = ((4, 4), (6, 6), (8, 8), (12, 12), (16, 16))

    with open(RESULTS_FILE, 'w') as f:
        f.write('num,cell_width,cell_height,block_width,block_height,' + \
                'width,height,accuracy\n')

    num = 1
    start_time = time.time()
    for cell_size in cell_sizes:
        for block_size in block_sizes:
            svm_time = time.time()

            logging.info("Block size: %s Cell size: %s", block_size, cell_size)
            logging.debug("Creating HOG features")
            w = (cell_size[0] - (target_width % cell_size[0])) + target_width
            h = (cell_size[1] - (target_height % cell_size[1])) + target_height

            if cell_size[0] * block_size[0] > w or \
               cell_size[1] * block_size[1] > h:
                logging.info("Skipping parameters")
                continue

            for i in range(1):
                do_hog(block_size=block_size, cell_size=cell_size,
                        char_size=(w, h),
                        window_size=(w+cell_size[0], h+cell_size[1]))

                logging.debug("Training SVM")
                SVM, accuracy = runSVM(HOG_DIR + 'train/', HOG_DIR + 'test/')
                logging.info("SVM training time: %f s", time.time() - svm_time)
                logging.info("SVM accuracy: %f", accuracy)

                with open(RESULTS_FILE, 'a') as f:
                    f.write('{},{},{},{},{},{},{},{}\n'.format(num,
                        cell_size[0], cell_size[1],
                        block_size[0], block_size[1],
                        w, h,
                        accuracy))
                num += 1

    logging.log("Total running time: %f s", time.time() - start_time)
