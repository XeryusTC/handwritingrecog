from __future__ import print_function
from unipath import Path
import sys
import xml.etree.ElementTree as ET
import cv2
import pickle
import logging
import logging.config

import general.preprocessor as prep
from general.hog import hog_xeryus
from recognize import cut_letters

sides = ('left', 'top', 'right', 'bottom')
MIN_WORD_LENGTH = 6
out_dir = Path('tmp/cuts/')

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

def find_cuts(img):
    window = 19
    img = cut_letters.removeWhitelines(word_img)
    if img is not None:
        if len(img[0]) <= window:
            cuts = [0, len(img[0])-1]
        else:
            hist = cut_letters.makeHist(img, window)
            cuts = cut_letters.findMaxima(hist)
        return cuts
    else:
        logger.warning('Image not good for classifying')
        return None

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python {} <image> <words>'.format(sys.argv[0]))
        sys.exit(1)

    img_file = sys.argv[1]
    word_file = sys.argv[2]

    logger.info('Loading pretrained models...')
    with open('tmp/svm.pickle', 'r') as f:
        svm = pickle.load(f)

    logger.info('Loading image and words file')
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = prep.preprocess(img)
    xml = ET.parse(word_file).getroot()

    out_dir.mkdir()
    for f in out_dir.walk('*'):
        f.remove()

    logger.info('Starting to create a split file')
    for sentence in xml:
        for word in sentence:
            text = word.get('text')
            print(text)
            #if '@' in text or len(text) < 6:
            if text != 'buton':
                continue # Skip short words

            # Get the word image
            rect = {side: int(word.get(side)) for side in sides}
            word_img = img[rect['top']:rect['bottom'], rect['left']:rect['right']]
            word_img = cut_letters.removeWhitelines(word_img)
            if word_img is None:
                logger.warning('Word not good for classifying: {}'.format(text))
                continue

            # Get the cuts
            cuts = find_cuts(word_img)
            if cuts is None:
                logger.warning('No cuts found in word')
                continue

            cv2.imwrite(out_dir + 'original.png', word_img)
            cuts_img = cut_letters.showCuts(word_img.copy(), cuts)
            cv2.imwrite(out_dir + 'cuts.png', cuts_img)

            i = 0
            for start in range(len(cuts)):
                for end in range(start+1, len(cuts)):
                    if not (15 < cuts[end] - cuts[start] < 100):
                        continue # Skip too small/too large sections
                    segment = word_img[:,cuts[start]:cuts[end]]
                    f = hog_xeryus(segment)
                    letter = svm.predict(f[:,0])
                    print('{}: {}:{} {}'.format(i, cuts[start], cuts[end], letter[0]))
                    cv2.imwrite(out_dir + str(i) + '-' + letter[0] + '.png', segment)

                    i += 1

            sys.exit(0)
