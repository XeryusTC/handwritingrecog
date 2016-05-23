### Packages
import create_segments
import sys
import logging

### Own modules
from general.hog import doHog
from general.pca import runPCA
from general.create_sets import create_sets
from train.svm import runSVM
from train.knn import runKNN

segmentDir = 'tmp/segments/'
featureDir = 'tmp/features/'
trainDir = featureDir + 'train/'
testDir = featureDir + 'test/'

for k in range(30):
    avgAcc = 0.0
    for i in range(10):
        doHog(segmentDir, featureDir, "xeryus")
        create_sets(featureDir)
        kNN, accuracy = runKNN(trainDir, testDir, "hog")

        avgAcc += accuracy
    print "Average accuracy for k =", k, ":\t", avgAcc/10
    avgAcc = 0
