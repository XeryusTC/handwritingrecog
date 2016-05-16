import numpy as np
import sys, os, cv2, glob, csv, shutil
from sklearn import svm

def hog(img):
    img = cv2.resize(img,(64, 128), interpolation = cv2.INTER_CUBIC)
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def doHog(imgDir, hogDir):
    print imgDir, hogDir
    if os.path.exists(hogDir):
        shutil.rmtree(hogDir)
    os.makedirs(hogDir)
    for subdir, dirs, files in os.walk(imgDir):
        for f in files:
            print f
            img = cv2.imread(os.path.join(subdir, f))
            hist = hog(img)

            print os.path.basename(os.path.normpath(subdir))

            histfile = open(hogDir + os.path.basename(os.path.normpath(subdir)) + '.csv', 'a')
            np.savetxt(histfile, np.reshape(hist, (1, len(hist))), delimiter=',', header=f)
            histfile.close()

if __name__ == '__main__':
    # Training
    doHog('imgs/trainImgs/', 'hogfeatures_small/trainHog/')
    filelist = glob.glob('hogfeatures_small/trainHog/*')
    labels = []
    trainData = []
    for letter in filelist:
        data = np.genfromtxt(letter, delimiter=',')
        for line in data:
            trainData.append(line)

        for label in range(len(data)):
            labels.append(os.path.splitext(os.path.basename(letter))[0])

    trainData = np.asarray(trainData)

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(trainData, labels)

    # Testing
    doHog('imgs/testImgs/', 'hogfeatures_small/testHog/')
    filelist = glob.glob('hogfeatures_small/testHog/*')
    accuracy = 0.0
    correct = 0.0
    false = 0.0

    for letter in filelist:
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
    accuracy = correct / (correct + false)
    print 'accuracy: ', accuracy
