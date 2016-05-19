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

    print "Commence training! Go walk your dog, because this is going to take a while..."
    clf = svm.SVC(decision_function_shape='ovo')
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

        for line in data:
            dec = clf.predict([line])
            print 'estimation: ', dec
            print 'actual: ', os.path.splitext(os.path.basename(letter))[0]
            if dec == os.path.splitext(os.path.basename(letter))[0]:
                print 'correct'
                correct += 1
            else:
                print 'incorrect'
                false += 1
            print "\n"
    accuracy = correct / (correct + false)
    return accuracy


if __name__ == '__main__':
    traindir = 'hogfeatures/train/'
    testdir = 'hogfeatures/test/'
    clf = train(traindir)
    accuracy = test(testdir)

    print 'accuracy: ', accuracy
