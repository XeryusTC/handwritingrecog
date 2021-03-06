import sys, os, cv2, logging
import numpy as np

def preprocess(img):
    """ Convert the given greyscale image to an otsu-thresholed one,
        remove any specks, and imperfections """

    logging.debug('Preproccesing the supplied image')
    try:
        # result = 255 - otsuContrast(img)
        result = otsu(img)
        result = speck_removal(result)
        result = morphology(result)
    except:
        logging.error("Unexpected error:", sys.exc_info()[0])
        raise
    else:
        logging.debug('Preprocessing completed')
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
    ots = otsu(img.copy())
    otsuC = otsuContrast(img.copy(), False)
    otsuNew = otsu(otsuC.copy())
    speck = speck_removal( ots.copy() )
    morph = morphology( ots.copy() )
    combination = morphology( speck.copy() )

    cv2.imwrite('tmp/original.png', img)
    cv2.imwrite('tmp/otsu.png', ots)
    cv2.imwrite('tmp/speck.png', speck)
    cv2.imwrite('tmp/morph.png', morph)
    cv2.imwrite('tmp/combination.png', combination)
    cv2.imwrite('tmp/otsuC.png', otsuC)
    cv2.imwrite('tmp/otsuNew.png', otsuNew)
