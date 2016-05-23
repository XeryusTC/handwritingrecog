import os, shutil
import numpy as np
import logging

def create_sets(hogDir):
    logging.info("Creating train and test set")

    ### Directory stuff
    if not os.path.exists(hogDir):
        print "You must first create HOG features"
        sys.exit(1)

    if os.path.exists(hogDir + 'train/'):
        shutil.rmtree(hogDir + 'train/')
    if os.path.exists(hogDir + 'test/'):
        shutil.rmtree(hogDir + 'test/')

    os.makedirs(hogDir + 'test/')
    os.makedirs(hogDir + 'train/')

    ### Divide dataset into train and test
    trainPercentage = 0.7
    features = np.load(hogDir + 'hog.npy')
    labels = np.load(hogDir + 'labels.npy')

    trainFeatures = []
    testFeatures = []
    trainLabels = []
    testLabels = []

    for sample in range(features.shape[0]):
        if np.random.random_sample() < trainPercentage:
            trainFeatures.append(features[sample])
            trainLabels.append(labels[sample])
        else:
            testFeatures.append(features[sample])
            testLabels.append(labels[sample])

    np.save(hogDir + 'train/hog', trainFeatures)
    np.save(hogDir + 'train/labels', trainLabels)
    np.save(hogDir + 'test/hog', testFeatures)
    np.save(hogDir + 'test/labels', testLabels)
