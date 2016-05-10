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
    if len(sys.argv) != 2 and sys.argv[1] not in ['KNMP', 'Stanford']:
        print "Usage: python %s <dataset>" % sys.argv[0]
        print "\tDataset should be either 'KNMP' or 'Stanford'"
        sys.exit(1)

    # create a clean temporary directory
    work_dir = Path("tmp")
    work_dir.rmtree()
    work_dir.mkdir()

    img_dir = Path(Path.cwd().ancestor(1), 'data/hwr_data/pages', sys.argv[1])
    ann_dir = Path(Path.cwd().ancestor(1), 'data/charannotations')
    images = img_dir.listdir('*.jpg')
    annotations = ann_dir.listdir(sys.argv[1] + '*.words')
    files = merge(images, annotations)

    for f in files:
        print "Preprocessing", str(f[0])
        p = Path("tmp", f[0].stem + '.ppm')
        img = cv2.imread(f[0], cv2.CV_LOAD_IMAGE_GRAYSCALE)
        img = preprocess(img)
        cv2.imwrite(p, img)

        preIm = pamImage.PamImage(p)
        e = ET.parse(f[1]).getroot()
        print "Segmenting {}...".format(f[0])
        segment(preIm, e, work_dir)

def merge(images, annotations):
    ret = []
    for img in images:
        for ann in annotations:
            if img.stem == ann.stem:
                ret.append((img, ann))
    return ret

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
