### Packages
import create_segments
import sys
import logging
import pickle

### Own modules
from general.hog import doHog
from general.pca import runPCA
from general.create_sets import create_sets
from train.svm import runSVM
from train.knn import runKNN

if __name__ == '__main__':
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--create_segments"):
        print sys.argv[1]
        print("Usage: python(2) %s (--create_segments)" % sys.argv[0])
        sys.exit(1)

    createSegments = False
    if len(sys.argv) == 2:
        createSegments = True

    ### Directories
    segmentDir = 'tmp/segments/'
    featureDir = 'tmp/features/'
    trainDir = featureDir + 'train/'
    testDir = featureDir + 'test/'

    ### First segment the training images
    if createSegments:
        logging.info("Creating segments of the images")
        create_segments.create_seg()

    ### Run hog over the segmented images
    ### Third argument for type of hog ("xeryus" or "other"), default = "xeryus"
    logging.info("Doing the HOG")
    doHog(segmentDir, featureDir, 'xeryus')

    ### Create random train and test sets from the hog features
    logging.info("Creating train and test set")
    create_sets(featureDir)

    ### Receive trained kNN (it also tests it on characters)
    logging.info("Running kNN")
    k = 10
    kNN = runKNN(trainDir, testDir, k)
    with open('tmp/knn.pickle', 'w') as f:
            pickle.dump(kNN, f)
