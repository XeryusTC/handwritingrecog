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
    # result = speck_removal(result)
    return result


def otsu(img):
    ret, result =  cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def speck_removal(img):
    # First convert to binary image
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,127,255,0)
    result, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contourThreshold = 50;
    for idx, contour in enumerate(contours):
        if contour.size < contourThreshold:
            cv2.drawContours(result, contours, idx, (255,255,255), 2, 8, hierarchy, 0)
            # cv2.drawContours(result, contours, idx, color, FILLED, 8, hierarchy );
            # cv2.drawContours(result, contours, idx, color, lineThickness, lineType, hierarchy, 0, Point() );
    return result

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0)
    otsu = otsu(img.copy())
    speck = speck_removal( otsu.copy() )
    cv2.imwrite('tmp/original.png', img)
    cv2.imwrite('tmp/otsu.png', otsu)
    cv2.imwrite('tmp/speck.png',speck)
    cv2.imwrite('tmp/otsu-speck.png', cv2.copyTo(otsu, speck))
