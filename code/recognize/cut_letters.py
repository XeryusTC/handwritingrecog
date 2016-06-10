import sys, os
import cv2, numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

def removeWhitelines(cropped_im):
    # Remove from the top
    while cropped_im.shape[1] > 0 and cropped_im.shape[0] > 0 \
            and min(cropped_im[0,:]) == 255:
        cropped_im = cropped_im[1:,:]
    # Remove from the bottom
    while cropped_im.shape[1] > 0 and cropped_im.shape[0] > 0 \
            and min(cropped_im[-1,:]) == 255:
        cropped_im = cropped_im[:-1,:]
    # Remove from the left
    while cropped_im.shape[1] > 0 and cropped_im.shape[0] > 0 \
            and min(cropped_im[:,0]) == 255:
        cropped_im = cropped_im[:,1:]
    # Remove from the right
    while cropped_im.shape[1] > 0 and cropped_im.shape[0] > 0 \
            and min(cropped_im[:,-1]) == 255:
        cropped_im = cropped_im[:,:-1]
    if cropped_im.shape[0] <= 5 or cropped_im.shape[1] <= 5:
        print "Discarding image"
        return None
    return cropped_im

def makeHist(img):
    hist = np.zeros(len(img[0]))
    for row in img:
        hist = [x + y for x, y in zip(hist, row)]
    # Smoothing
    hist = savgol_filter(hist, 19, 3)
    return hist

def findMaxima(hist):
    maxima = [0, len(hist)-1]
    for index, value in enumerate(hist):
        if not index < 10 and not index > len(hist) - 10:
            if value > hist[index - 1] and value > hist[index + 1]:
                maxima.append(index)
    return sorted(maxima)

def showCuts(img, cuts):
    # Draw vertical lines at the places of the cuts
    for row in img:
        for cut in cuts:
            row[cut] = 0
    return img

def cutLetters(img):
    img = removeWhitelines(img)
    if not img is None:
        hist = makeHist(img)
        cuts = findMaxima(hist)
        return cuts
    else:
        print "image not good for classifying"
        return None

    # Now do magic stuff with the statistics from Xeryus' function to get the final letters
    # ...

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python %s image.ppm" % sys.argv[0]
        sys.exit(0)

    name = os.path.splitext(sys.argv[1])[0]
    img = cv2.imread(sys.argv[1], 0)
    img = removeWhitelines(img)
    hist = makeHist(img)
    cuts = findMaxima(hist)
    sliced = showCuts(img, cuts)

    # Save created images
    plt.plot(hist)
    plt.savefig(name + '.png')
    cv2.imwrite(name + 'sliced.png', sliced)
