import create_segments
import sys
from general.hog import doHog
from train.svm import svm

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['KNMP', 'Stanford']:
        print("Usage: python %s <dataset>" % sys.argv[0])
        print("\tDataset should be either 'KNMP' or 'Stanford'")
        sys.exit(1)

    # First segment the training images
    create_segments.create(sys.argv[1])

    # Run hog over the segmented images
    # Third optonial argument for type of hog ("xeryus" or "alternative"), default = "xeryus"
    doHog('tmp/segments', 'tmp/hog_features')

    # Receive trained svm (it also tests it on characters)
    SVM = svm('tmp/hog_features/train', 'tmp/hog_features/test')
