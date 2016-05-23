import sys, os, cv2, glob
import numpy as np
from sklearn import neighbors
import logging

def train(traindir, featuretype):
    logging.debug("Training kNN!")

    n_neighbors = 5
    weights = 'uniform' # Other: 'distance'
    if featuretype == "hog":
        trainData = np.load(traindir + 'hog.npy')
    else:
        trainData = np.load(traindir + 'pca.npy')
    trainLabels = np.load(traindir + 'labels.npy')

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(trainData, trainLabels)

    return clf

def test(testdir, clf, featuretype):
    logging.debug("Testing kNN!")

    accuracy = 0.0
    correct = 0.0
    false = 0.0
    if featuretype == "hog":
        testData = np.load(testdir + 'hog.npy')
    else:
        testData = np.load(testdir + 'pca.npy')
    labels = np.load(testdir + 'labels.npy')

    label = 0
    for line in testData:
        dec = clf.predict([line])
        # print 'estimation: ', dec
        # print 'actual: ', labels[label]
        if dec == labels[label]:
            correct += 1
        else:
            false += 1
        label += 1

    accuracy = correct / (correct + false)
    return accuracy

def runKNN(traindir, testdir, featuretype = "hog"):
    logging.info("Running kNN")
    clf = train(traindir, featuretype)
    accuracy = test(testdir, clf, featuretype)
    return clf, accuracy
