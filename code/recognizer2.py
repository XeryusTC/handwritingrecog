import sys
import os
import cv2
import toolbox.pamImage as pamImage
import toolbox.croplib as croplib
import xml.etree.ElementTree as ET

def main():
    if len(sys.argv) != 4:
        print "Usage: python %s image.ppm input.words output.words" % sys.argv[0]
        sys.exit(1)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    in_file = sys.argv[1]
    in_words = sys.argv[2]
    out_words = sys.argv[3]

    print "Preprocessing..."
<<<<<<< HEAD
    img = cv2.imread(in_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
=======
    img = cv2.imread(in_file, cv2.IMREAD_GRAYSCALE);
>>>>>>> 218a591e0cf266df0acef9d3b8dd21fb1b413ac9
    result = preprocess(img)
    cv2.imwrite("tmp/preprocessed.ppm", result)

    preIm = pamImage.PamImage("tmp/preprocessed.ppm")
    e = ET.parse(in_words).getroot()
    words = segment(preIm, e)
    tree = ET.ElementTree(words)
    tree.write(out_words)

def preprocess(img, stretch=True):
    # contrast stretching
    if stretch:
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # Otsu thresholding
    thresh, result = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU);
<<<<<<< HEAD
    print "Thresholded at", thresh
    cv2.imwrite("tmp/preprocessed.ppm", result)

    # Remove specks
    temp = result.copy()
    contourThreshold = 50
    lineThickness = 2
    lineType = 8 # 4, 8 or CV_AA

    for i in range(3):
        temp, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for vector in contours:
        if vector.size() > contourThreshold:
            contours.remove(vector)

    cv2.drawContours(temp, contours, -1, (255,255,255), lineThickness)

    result = result - temp;
    cv2.imwrite("tmp/preprocessed2.ppm", result)

=======
    result = 255 - result
>>>>>>> 218a591e0cf266df0acef9d3b8dd21fb1b413ac9
    return result

def segment(img, words):
    # Parse the given words file to find the word sections
    for sentence in words:
        left = int(sentence.get('left'))
        top = int(sentence.get('top'))
        right = int(sentence.get('right'))
        bottom = int(sentence.get('bottom'))
        cropped_im = croplib.crop(img, left, top, right, bottom)
        cropped_im.thisown = True # Makes Python clean up the C++ object
        cropped_im.save('tmp/sentence{}.ppm'.format(sentence.get('no')))

        # Individual words
        for word in sentence:
            left = int(word.get('left'))
            top = int(word.get('top'))
            right = int(word.get('right'))
            bottom = int(word.get('bottom'))
            cropped_im = croplib.crop(img, left, top, right, bottom)
            cropped_im.thisown = True # Makes Python clean up the C++ object later
            cropped_im.save('tmp/sengtence{}_word{}.ppm'.format(sentence.get('no'), word.get('no')))

            # classify the word
            word.set('text', 'dunno')
            print "Set text: %s" % word.get('text')
    return words

if __name__ == '__main__':
    main()
