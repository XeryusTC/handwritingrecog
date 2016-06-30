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
    if len(sys.argv) != 2 or sys.argv[1] not in ['KNMP', 'Stanford']:
        print("Usage: python %s <dataset> <hogtype> <featuretype>" % sys.argv[0])
        print("\tDataset should be either 'KNMP' or 'Stanford'")
        sys.exit(1)

    ### Directories
    segmentDir = 'tmp/segments/'
    featureDir = 'tmp/features/'
    trainDir = featureDir + 'train/'
    testDir = featureDir + 'test/'

    ### First segment the training images
    # logging.info("Creating segments of the images")
    # create_segments.create(sys.argv[1])

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
    kNN, accuracy = runKNN(trainDir, testDir, k)
    with open('knn.pickle', 'w') as f:
        pickle.dump(kNN, f)

    print 'Accuracy for kNN: ', accuracy
