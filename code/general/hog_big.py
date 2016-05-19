import sys, os, cv2
import numpy as np

if __name__ == '__main__':
    rootdir = 'imgs/tmp/'
    if not os.path.exists('hogfeatures'):
        os.makedirs('hogfeatures')
    if not os.path.exists('hogfeatures/test'):
        os.makedirs('hogfeatures/test')
    if not os.path.exists('hogfeatures/train'):
        os.makedirs('hogfeatures/train')

    for subdir, dirs, files in os.walk(rootdir):
        train = 0
        for f in files:
            # Read image and resize it
            print os.path.join(subdir, f)
            image = cv2.imread(os.path.join(subdir, f))
            img = cv2.resize(image,(64, 128), interpolation = cv2.INTER_CUBIC)

            # Parameters for HOG
            winSize = (32,32)
            blockSize = (8,8)
            blockStride = (4,4)
            cellSize = (4,4)
            nbins = 9
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = 2.0000000000000001e-01
            gammaCorrection = 0
            nlevels = 64

            winStride = (16, 16)
            padding = (0, 0)

            # Run HOG
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                    histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
            hist = hog.compute(img,winStride,padding)

            # Store
            if train < 20:
                histfile = open('hogfeatures_big/test/' + os.path.basename(os.path.normpath(subdir)) + '.dat', 'a')
            else:
                histfile = open('hogfeatures_train/train/' + os.path.basename(os.path.normpath(subdir)) + '.dat', 'a')
            np.savetxt(histfile, np.reshape(hist, (1, len(hist))), delimiter=',', header=f)
            histfile.close()
            train += 1
