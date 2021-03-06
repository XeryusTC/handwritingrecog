import sys, os, cv2, glob
import numpy as np
from sklearn import svm
import logging

def train(traindir, featuretype):
    logging.debug("Training svm!")

    if featuretype == "hog":
        trainData = np.load(traindir + 'hog.npy')
    else:
        trainData = np.load(traindir + 'pca.npy')
    labels = np.load(traindir + 'labels.npy')

    # One vs all approach
    clf = svm.LinearSVC()

    # One vs one approach
    # clf = svm.SVC(decision_function_shape='ovo')

    clf.fit(trainData, labels)
    return clf


def test(testdir, clf, featuretype):
    logging.debug("Testing svm!")

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

def runSVM(traindir, testdir, featuretype = "hog"):
    logging.info("Training SVM")
    clf = train(traindir, featuretype)
    logging.info("Testing SVM")
    accuracy = test(testdir, clf, featuretype)
    return clf, accuracy
