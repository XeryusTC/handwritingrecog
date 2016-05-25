# Packages
import sys, os, cv2, uuid, logging, csv
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
    # create a clean temporary directory
    work_dir = Path("tmp")
    work_dir.mkdir()

    lexicon = {}

    # Find all the annotated pages in the dataset
    ann_dir = Path(Path.cwd().ancestor(1), 'data/charannotations')
    logging.info('annd dir %s' % ann_dir)
    annotations = ann_dir.listdir(dataset + '*.words')

    for f in annotations:
        # Preprocess
        logging.info("processing %s", str(f))
        # Segment
        annotation = ET.parse(f).getroot()
        for word in annotation.iter('Word'):
            text = word.get('text')

            # Add word to lexicon
            logging.info("found word %s", text)
            if lexicon.has_key(text):
                lexicon[text] = lexicon[text] + 1
            else :
                lexicon[text] = 1

    print lexicon
    lexicon_path = Path(work_dir, "lexicon_" + str(dataset) + ".csv")
    w = csv.writer(open(lexicon_path, "w"))
    for key, val in lexicon.items():
        w.writerow([key, val])
if __name__ == '__main__':
    main()
