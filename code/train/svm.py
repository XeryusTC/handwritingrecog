import sys, os, cv2, glob
import numpy as np
from sklearn import svm

def train(traindir):
    print "Training..."

    filelist = glob.glob(traindir + '*')
    labels = []
    trainData = []
    for letter in filelist:
        print "Prepping training data for letter: ", os.path.splitext(os.path.basename(letter))[0]
        data = np.genfromtxt(letter, delimiter=',')
        for line in data:
            trainData.append(line)

        for label in range(len(data)):
            labels.append(os.path.splitext(os.path.basename(letter))[0])

    trainData = np.asarray(trainData)
    
    print "Commence training!"
    
    # One vs all approach
    clf = svm.LinearSVC()
    
    # One vs one approach
    # clf = svm.SVC(decision_function_shape='ovo')
    
    clf.fit(trainData, labels)
    
    return clf


def test(testdir):
    print "Testing..."

    filelist = glob.glob(testdir + '*')
    accuracy = 0.0
    correct = 0.0
    false = 0.0
    for letter in filelist:
        print "testing letter: ", letter
        data = np.genfromtxt(letter, delimiter=',')
        
        if len(data.shape) > 1:
            for line in data:
                dec = clf.predict([line])
                print 'estimation: ', dec
                print 'actual: ', os.path.splitext(os.path.basename(letter))[0], '\n'
                if dec == os.path.splitext(os.path.basename(letter))[0]:
                    correct += 1
                else:
                    false += 1
                print "\n"
    accuracy = correct / (correct + false)
    return accuracy


def svm(traindir, testdir):
    clf = train(traindir)
    accuracy = test(testdir)

    print 'accuracy: ', accuracy
    
    return clf
