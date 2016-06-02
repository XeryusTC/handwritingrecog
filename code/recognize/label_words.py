import sys, os, logging, shutil
import xml.etree.ElementTree as ET
from unipath import Path
import cv2

from . import cut_letters

sides = ('left', 'top', 'right', 'bottom')

sys.path.insert(1, os.path.join(sys.path[0], '..'))

class Recognizer(object):
    def __init__(self, sentence_dir, word_dir, words, img):
        # Reset directories
        sentence_dir.rmtree()
        sentence_dir.mkdir()
        self.sentence_dir = sentence_dir
        word_dir.rmtree()
        word_dir.mkdir()
        self.word_dir = word_dir

        self.words = words
        self.img = img

    def next_word(self):
        for sentence in self.words:
            for word in sentence:
                rect = {side: int(word.get(side)) for side in sides}
                word_img = self.img[rect['top']:rect['bottom'],
                    rect['left']:rect['right']]
                word_img = cut_letters.removeWhitelines(word_img)
                if word_img is None:
                    continue

                yield word, word_img

    def find_cuts(self, word_img):
        hist = cut_letters.makeHist(word_ig)
        cuts = cut_letters.findMaximma(hist)
        return cuts

    def recognize(self, word_img, cuts):
        return "make this"


def getWords(img, xml, sentenceDir, wordDir):
    sentenceDir.rmtree()
    sentenceDir.mkdir()
    wordDir.rmtree()
    wordDir.mkdir()

    # Parse the given words (xml) file to find the word sections
    logging.info("Cropping words")
    for sentence in xml:
        rect = {side: int(sentence.get(side)) for side in sides}
        no = sentence.get('no')
        cropped_im = img[rect['top']:rect['bottom'],
            rect['left']:rect['right']]
        cv2.imwrite(sentenceDir.child('sentence' + no + '.ppm'), cropped_im)

        # Now to the individual words
        for word in sentence:
            rect = {side: int(word.get(side)) for side in sides}
            word_no = word.get('no')
            cropped_im = img[rect['top']:rect['bottom'],
                rect['left']:rect['right']]
            cv2.imwrite(wordDir.child(no + '_word' + word_no + '.ppm'),
                cropped_im)

            # classify the word
            # word.set('text', 'dunno')

    # Return the new filled in tree
    tree = ET.ElementTree(xml)
    logging.info("Words cropped")
    return tree
    # shutil.rmtree('tmp')
