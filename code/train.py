import create_segments
import sys
from general.hog import doHog
from general.pca import runPCA
from train.svm import runSVM

if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in ['KNMP', 'Stanford']:
        print("Usage: python %s <dataset> <hogtype> <featuretype>" % sys.argv[0])
        print("\tDataset should be either 'KNMP' or 'Stanford'")
        print("\tHogtype should be either 'xeryus' or 'other'")
        print("\tFeaturetype should be either 'hog' or 'pca'")
        sys.exit(1)

    ### First segment the training images
    #create_segments.create(sys.argv[1])

    ### Run hog over the segmented images
    ### Third argument for type of hog ("xeryus" or "other"), default = "xeryus"
    doHog('tmp/segments/', 'tmp/features/', sys.argv[2])

    ### Get the principal components from the hog_features
    runPCA('tmp/features/')

    ### Receive trained svm (it also tests it on characters)
    ### Third argument either "hog" or "pca", default = "hog"
    SVM,accuracy = runSVM('tmp/features/train/', 'tmp/features/test/', sys.argv[3])
    print 'Accuracy:', accuracy
