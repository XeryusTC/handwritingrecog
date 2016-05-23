# Packages
import sys, os, cv2, uuid, logging
import logging.config
import numpy as np
from unipath import Path, DIRS_NO_LINKS
from scipy.cluster.vq import whiten, kmeans2
import xml.etree.ElementTree as ET

# Own modules
from general.preprocessor import preprocess

logging.config.fileConfig('logging.conf')

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['KNMP', 'Stanford']:
        print("Usage: python %s <dataset>" % sys.argv[0])
        print("\tDataset should be either 'KNMP' or 'Stanford'")
        sys.exit(1)
    create(sys.argv[1])

def create(dataset):
    logging.info("Creating segments of the images")
    
    # create a clean temporary directory
    work_dir = Path("tmp")
    work_dir.rmtree()
    work_dir.mkdir()

    # Find all the pages in the dataset
    img_dir = Path(Path.cwd().ancestor(1), 'data/hwr_data/pages', dataset)
    ann_dir = Path(Path.cwd().ancestor(1), 'data/charannotations')
    images = img_dir.listdir('*.jpg')
    annotations = ann_dir.listdir(dataset + '*.words')
    files = merge(images, annotations)

    # Create character segmentations
    for f in files:
        # Preprocess
        logging.info("Preprocessing %s", str(f[0]))
        pagesPathFolder = Path(work_dir, 'pages')
        pagesPathFolder.mkdir()
        pagePath = Path(pagesPathFolder, f[0].stem + '.ppm')
        img = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE)
        img = preprocess(img)
        cv2.imwrite(pagePath, img)

        # Segment
        segmentPathFolder = Path(work_dir, 'segments')
        segmentPathFolder.mkdir()
        e = ET.parse(f[1]).getroot()
        logging.info("Segmenting %s", str(f[0]))
        segment(img, e, segmentPathFolder)

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

                rect = {side: int(char.get(side)) for side in sides}
                # Correct for swapped coordinates
                if rect['top'] > rect['bottom']:
                    rect['top'], rect['bottom'] = rect['bottom'], rect['top']
                if rect['left'] > rect['right']:
                    rect['left'], rect['right'] = rect['right'], rect['left']
                cropped_im = img[rect['top']:rect['bottom'], rect['left']:rect['right']]

                # Remove rows from the top if they're white
                while cropped_im.shape[0] > 0 and min(cropped_im[0,:]) == 255:
                    cropped_im = cropped_im[1:,:]
                # Remove from the bottom
                while cropped_im.shape[0] > 0 and min(cropped_im[-1,:]) == 255:
                    cropped_im = cropped_im[:-1,:]
                # Remove from the left
                while cropped_im.shape[1] > 0 and min(cropped_im[:,0]) == 255:
                    cropped_im = cropped_im[:,1:]
                # Remove from the right
                while cropped_im.shape[1] > 0 and min(cropped_im[:,-1]) == 255:
                    cropped_im = cropped_im[:,:-1]
                cv2.imwrite(f, cropped_im)

if __name__ == '__main__':
    main()
