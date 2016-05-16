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

    result = otsu(img)
    result = speck_removal(result)
    result = morphology(result)
    # cv2.imwrite('tmp/preprocessed.ppm', result)

    return result;

def otsu(img):
    ret, result =  cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def otsuContrast(img, stretch = True):
    if stretch:
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    thresh, result = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU)
    return result

def speck_removal(img):
    copy = img.copy()
    result, contours, hierarchy = cv2.findContours(copy,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contourThreshold = 30;
    mask = np.ones(img.shape[:2], dtype="uint8")
    for idx, contour in enumerate(contours):
        if contour.size <= contourThreshold:
            cv2.drawContours(mask, contours, idx, (255,255,255), 2, 8, hierarchy, 0)
    # Remove the mask from the img ( make contours white)
    return np.maximum(img, mask)

def morphology(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations = 1)
    img = cv2.erode(img, kernel, iterations = 1)
    return img

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0)
    otsu = otsu(img.copy())
    speck = speck_removal( otsu.copy() )
    morph = morphology( otsu.copy() )
    combination = morphology( speck.copy() )

    cv2.imwrite('tmp/original.png', img)
    cv2.imwrite('tmp/otsu.png', otsu)
    cv2.imwrite('tmp/speck.png', speck)
    cv2.imwrite('tmp/morph.png', morph)
    cv2.imwrite('tmp/combination.png', combination)
