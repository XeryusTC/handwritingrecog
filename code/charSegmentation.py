# Packages
import sys, os, cv2
import numpy as np
from unipath import Path, DIRS_NO_LINKS
from scipy.cluster.vq import whiten, kmeans2
import xml.etree.ElementTree as ET

# Own modules
from modules.segment import segment
from modules.preprocessor import preprocess

def merge(images, annotations):
    ret = []
    for img in images:
        for ann in annotations:
            if img.stem == ann.stem:
                ret.append((img, ann))
    return ret

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
    files = merge(images, annotations)

    # Create character segmentations
    for f in files:
        # Preprocess
        print "Preprocessing", str(f[0])
        p = Path("tmp", f[0].stem + '.ppm')
        img = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE)
        img = preprocess(img)
        # cv2.imwrite(p, img)

        # Segment
        e = ET.parse(f[1]).getroot()
        print "Segmenting {}...".format(f[0])
        segment(img, e, work_dir)

if __name__ == '__main__':
    main()
