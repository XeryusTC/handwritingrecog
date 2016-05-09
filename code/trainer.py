import sys
import os
import cv2
import uuid
from unipath import Path

import toolbox.pamImage as pamImage
import toolbox.croplib as croplib
import xml.etree.ElementTree as ET

from recognizer2 import preprocess

def main():
    if len(sys.argv) != 3:
        print "Usage: python %s image.ppm input.words" % sys.argv[0]
        sys.exit(1)

    # create a clean temporary directory
    work_dir = Path("tmp")
    work_dir.rmtree()
    work_dir.mkdir()

    in_image = sys.argv[1]
    in_words = sys.argv[2]
    process_image(in_image, in_words, work_dir)

def process_image(in_image, in_words, work_dir):
    print "Preprocessing {}...".format(in_image)
    img = cv2.imread(in_image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    result = preprocess(img)

    preIm = pamImage.PamImage("tmp/preprocessed.ppm")
    e = ET.parse(in_words).getroot()
    chars = segment(preIm, e, work_dir)
    return chars

def segment(img, annotation, work_dir):
    sides = ['left', 'top', 'right', 'bottom']
    # Parse the given sentences
    for sentence in annotation:
        for word in sentence:
            for char in word:
                c = char.get('text')
                cdir = Path(work_dir, c)
                cdir.mkdir()
                f = Path(cdir, str(uuid.uuid1()) + '.ppm')

                rect = [int(char.get(side)) for side in sides]
                cropped_im = croplib.crop(img, *rect)
                cropped_im.thisown = True # Makes python clean up the C++ object later
                cropped_im.save(str(f))

if __name__ == '__main__':
    main()
