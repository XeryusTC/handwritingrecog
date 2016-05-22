import sys, os, cv2, glob, shutil
import numpy as np
from sklearn.decomposition import PCA

def runPCA(hogDir):
    ### Directory stuff
    if not os.path.exists(hogDir):
        print "You must first create HOG features"
        sys.exit(1)

    trainData = np.load(hogDir + 'train/hog.npy')
    testData = np.load(hogDir + 'test/hog.npy')

    print "shape trainData: ", trainData.shape
    print "shape testData: ", testData.shape

    data = np.append(trainData, testData, axis = 0)
    print "data shape: ", data.shape

    ### Fit the pca model
    print "Fitting the pca model"
    pca = PCA(n_components = min(testData.shape[0], trainData.shape[0], testData.shape[1], trainData.shape[1]))
    pca.fit(data)
    trainData = pca.fit_transform(trainData)
    testData = pca.fit_transform(testData)

    # trainData = np.asarray(trainData)
    print "shape trainData: ", trainData.shape
    print "shape testData: ", testData.shape

    ### Store the labels and the array with transformed feature vectors
    np.save(hogDir + 'train/pca', trainData)
    np.save(hogDir + 'test/pca', testData)
