import sys
import os
import cv2
import uuid
from unipath import Path, DIRS_NO_LINKS

import toolbox.pamImage as pamImage
import toolbox.croplib as croplib
import xml.etree.ElementTree as ET

from recognizer2 import preprocess

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['KNMP', 'Stanford']:
        print "Usage: python %s <dataset>" % sys.argv[0]
        print "\tDataset should be either 'KNMP' or 'Stanford'"
        sys.exit(1)

    # create a clean temporary directory
    work_dir = Path("tmp")
    work_dir.rmtree()
    work_dir.mkdir()

    # Find all the pages in the dataset
    img_dir = Path(Path.cwd().ancestor(1), 'data/hwr_data/pages', sys.argv[1])
    ann_dir = Path(Path.cwd().ancestor(1), 'data/charannotations')
    images = img_dir.listdir('*.jpg')
    annotations = ann_dir.listdir(sys.argv[1] + '*.words')
    files = merge(images, annotations)[:1]

    maxwidth = 0
    maxheight = 0
    # Create character segmentations
    for f in files:
        print "Preprocessing", str(f[0])
        p = Path("tmp", f[0].stem + '.ppm')
        img = cv2.imread(f[0], cv2.CV_LOAD_IMAGE_GRAYSCALE)
        img = preprocess(img)
        cv2.imwrite(p, img)

        preIm = pamImage.PamImage(p)
        e = ET.parse(f[1]).getroot()
        print "Segmenting {}...".format(f[0])
        width, height = segment(preIm, e, work_dir)

        # keep track of the maximum width and height
        if width > maxwidth:
            maxwidth = width
        if height > maxheight:
            maxheight = height

    # Make the maxwidth and maxheight a multiple of 16
    if maxwidth % 16 != 0:
        maxwidth += 16 - (maxwidth % 16)
    if maxheight % 16 != 0:
        maxheight += 16 - (maxheight % 16)

    # Find all letters
    print "Building HOG descriptor from letters..."
    letters = work_dir.listdir(filter=DIRS_NO_LINKS)
    data = []
    hog = cv2.HOGDescriptor(_winSize=(32,32), _blockSize=(16,16),
        _blockStride=(2,2), _cellSize=(8,8),_nbins=9)
    for letter in letters:
        examples = letter.listdir(pattern='*.ppm')
        # Find the HOG descriptor for each example image
        for example in examples:
            # load example and make it a standard size
            img = cv2.imread(example, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            left = (maxwidth - img.shape[1]) / 2
            top = (maxheight - img.shape[0]) / 2
            right = maxwidth - img.shape[1] - left
            bottom = maxheight - img.shape[0] - top
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

            # compute the actual HOG descriptor
            desc = hog.compute(img)
            data.append(desc)

def merge(images, annotations):
    ret = []
    for img in images:
        for ann in annotations:
            if img.stem == ann.stem:
                ret.append((img, ann))
    return ret

def segment(img, annotation, work_dir):
    sides = ['left', 'top', 'right', 'bottom']
    maxwidth = 0
    maxheight = 0
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

                # calculate the width and height and update the maxima
                if abs(rect[0] - rect[2]) > maxwidth:
                    maxwidth = abs(rect[0] - rect[2])
                if abs(rect[1] - rect[3]) > maxheight:
                    maxheight = abs(rect[1] - rect[3])
    return maxwidth, maxheight

if __name__ == '__main__':
    main()
