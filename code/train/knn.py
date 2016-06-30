import sys, os, cv2, glob
import numpy as np
from sklearn import neighbors
import logging

def train(traindir, k):
    logging.info("Training kNN!")

    n_neighbors = k
    weights = 'uniform' # Other: 'distance'
    trainData = np.load(traindir + 'hog.npy')
    trainLabels = np.load(traindir + 'labels.npy')

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, n_jobs=-1)#, metric="euclidean")
    clf.fit(trainData, trainLabels)

    logging.info("Done Training!")
    return clf

def test(traindir, testdir, clf):
    logging.info("Testing kNN!")

    accuracy = 0.0
    correct = 0.0
    false = 0.0
    testData = np.load(testdir + 'hog.npy')
    labels = np.load(testdir + 'labels.npy')
    trainLabels = np.load(traindir + 'labels.npy')

    label = 0
    classes = sorted(set(trainLabels))
    nrOptions = [0, 0, 0]
    for line in testData:
        dec = clf.predict([line])
        logging.info('Predicted class %s' % dec)
        logging.info('Correct class: %s' % labels[label])

        probs = clf.predict_proba([line])

        # logging.info('classes= %s' % classes)
        options = -1
        for idx, val in enumerate(probs[0]):
            if val > 0.3:
                print 'prob: %s, class: %s' % (val, classes[idx])
                options += 1
        nrOptions[options] += 1
        # print 'estimation: ', dec
        # print 'actual: ', labels[label]
        if dec == labels[label]:
            correct += 1
        else:
            false += 1
        label += 1
    logging.info('Correct: %s, False: %s' % (correct, false))
    logging.info('Option times: %s %s %s' % (nrOptions[0], nrOptions[1], nrOptions[2]))
    accuracy = correct / (correct + false)
    logging.info("Probs")
    logging.info("Done Testing!")
    return accuracy

def getPredictions(clf, featureVector, classes):
    predictions = {}
    probs = clf.predict_proba(featureVector)
    for idx, val in enumerate(probs[0]):
        if val != 0:
            predictions[classes[idx]] = val
            #  print 'prob: %s, class: %s' % (val, classes[idx])
    return predictions

def runKNN(traindir, testdir, k):
    clf = train(traindir, k)
    # accuracy = test(traindir, testdir, clf)
    return clf, accuracy
