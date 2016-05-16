# Packages
import sys, os, cv2
import numpy as np

if __name__ == '__main__':
    rootdir = 'tmp2/'
    if not os.path.exists('hogfeatures'):
        os.makedirs('hogfeatures')

    for subdir, dirs, files in os.walk(rootdir):
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
            print len(hist)
            print np.transpose(hist)

            # Store
            #np.savez('hogfeatures/' + os.path.basename(os.path.normpath(subdir)), hist)
            histfile = open('hogfeatures/' + os.path.basename(os.path.normpath(subdir)) + '.csv', 'a')
            np.savetxt(histfile, np.transpose(hist), delimiter=',', header=f)
            histfile.close()
