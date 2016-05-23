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

if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in ['KNMP', 'Stanford']:
        print("Usage: python %s <dataset> <hogtype> <featuretype>" % sys.argv[0])
        print("\tDataset should be either 'KNMP' or 'Stanford'")
        print("\tHogtype should be either 'xeryus' or 'other'")
        print("\tFeaturetype should be either 'hog' or 'pca'")
        sys.exit(1)

    ### Directories
    segmentDir = 'tmp/segments/'
    featureDir = 'tmp/features/'
    trainDir = featureDir + 'train/'
    testDir = featureDir + 'test/'

    ### First segment the training images
    # create_segments.create(sys.argv[1])

    ### Run hog over the segmented images
    ### Third argument for type of hog ("xeryus" or "other"), default = "xeryus"
    doHog(segmentDir, featureDir, sys.argv[2])

    ### Create random train and test sets from the hog features
    create_sets(featureDir)

    ### Get the principal components from the hog_features
    runPCA(featureDir)

    ### Receive trained svm (it also tests it on characters)
    ### Third argument either "hog" or "pca", default = "hog"
    SVM, accuracy = runSVM(trainDir, testDir, sys.argv[3])

    ### Receive trained kNN (it also tests it on characters)
    ### Third argument either "hog" or "pca", default = "hog"
    kNN, accuracy2 = runKNN(trainDir, testDir, sys.argv[3])

    print 'Accuracy for SVM: ', accuracy
    print 'Accuracy for kNN: ', accuracy2
