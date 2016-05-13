import sys
import os
import cv2
import numpy as np
import toolbox.pamImage as pamImage
import toolbox.croplib as croplib
from matplotlib import pyplot as plt

def preprocess(img):
    """ Convert the given greyscale image to an otsu-thresholed one,
        remove any specks, and imperfections """

    # Otsu
    result = otsu(img)
    result = speck_removal(result)
    return result


def otsu(img):
    ret, result =  cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def speck_removal(img):
    # First convert to binary image
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    copy = img.copy()
    result, contours, hierarchy = cv2.findContours(copy,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    # contourThreshold = 50000;
    mask = np.ones(img.shape[:2], dtype="uint8")
    for idx, contour in enumerate(contours):
        # if contour.size <= contourThreshold:
        cv2.drawContours(mask, contours, idx, (255,255,255), 2, 8, hierarchy, 0)
    # Remove the mask from the img ( make contours white)
    return np.maximum(img, mask)

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0)
    otsu = otsu(img.copy())
    speck = speck_removal( otsu.copy() )
    speck2 = speck_removal2( otsu.copy() )
    cv2.imwrite('tmp/original.png', img)
    cv2.imwrite('tmp/otsu.png', otsu)
    cv2.imwrite('tmp/speck.png',speck)
