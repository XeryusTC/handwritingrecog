import sys, os
import cv2, numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

def makeHist(img):
    hist = np.zeros(len(img[0]))
    for row in img:
        hist = [x + y for x, y in zip(hist, row)]
    # Smoothing
    hist = savgol_filter(hist, 15, 3)
    return hist

def findMaxima(hist):
    maxima = []
    for index, value in enumerate(hist):
        if not index == 0 and not index == len(hist) - 1:
            if value > hist[index - 1] and value > hist[index + 1]:
                maxima.append(index)
    return maxima

def showCuts(img, cuts):
    # Draw vertical lines at the places of the cuts
    for row in img:
        for cut in cuts:
            row[cut] = 0
    return img

def cutLetters(img):
    hist = makeHist(img)
    cuts = findMaxima(hist)

    # Now do magic stuff with the statistics from Xeryus' function to get the final letters
    # ...

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python %s image.ppm" % sys.argv[0]

    name = os.path.splitext(sys.argv[1])[0]
    img = cv2.imread(sys.argv[1], 0)

    hist = makeHist(img)
    cuts = findMaxima(hist)
    sliced = showCuts(img, cuts)

    # Save created images
    plt.plot(hist)
    plt.savefig('tmp/' + name + '.png')
    cv2.imwrite('tmp/' + name + 'sliced.png', sliced)
