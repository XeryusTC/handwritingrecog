import sys, os, logging, shutil
import xml.etree.ElementTree as ET
from unipath import Path
import cv2
import pickle
import logging

from . import cut_letters
from general import hog

sides = ('left', 'top', 'right', 'bottom')

sys.path.insert(1, os.path.join(sys.path[0], '..'))

class Recognizer(object):
    def __init__(self, sentence_dir, word_dir, words, img):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        # Reset directories
        self.sentence_dir = sentence_dir
        self.word_dir = word_dir

        self.words = words
        self.img = img

        self.logger.info('Loading pretrained models...')
        with open('svm-pretrained.pickle', 'r') as f:
            self.svm = pickle.load(f)
        with open('knn.pickle', 'r') as f:
            self.knn = pickle.load(f)
        self.logger.info('Models loaded')

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
        img = cut_letters.removeWhitelines(word_img)
        if img is not None:
            hist = cut_letters.makeHist(img)
            cuts = cut_letters.findMaxima(hist)
            return cuts
        else:
            print "image not good for classifying"
            return None

    def getPredictions(self, clf, featureVector, classes):
        predictions = {}
        probs = clf.predict_proba(featureVector)
        for idx, val in enumerate(probs[0]):
            if val != 0:
                predictions[classes[idx]] = val
                #  print 'prob: %s, class: %s' % (val, classes[idx])
        return predictions


    def recognize(self, word_img, cuts, lexicon, stateProbs, transProbs, classes):
        text = ""
        hypotheses = {} # A graph that keeps track of all possible words
        for start in range(len(cuts)):
            hypotheses[cuts[start]] = []
            for end in range(start, len(cuts)):
                if not 10 <= (cuts[end] - cuts[start]) < 80:
                    continue
                window = word_img[:,cuts[start]:cuts[end]]
                window = cut_letters.removeWhitelines(window)
                if not window is None:
                    f = hog.hog_xeryus(window).reshape(1, -1)
                    l = self.getPredictions(self.knn, f, classes)
                    # l = self.knn.predict(f)
                    # l = self.svm.predict(f)
                    text = text + l[0]
                    hypotheses[cuts[start]].append((l[0], cuts[end]))
                else:
                    continue

        # Turn the hypotheses tree into a list of candidates
        candidates = self._hypotheses_graph_to_candidates(hypotheses)
        # print candidates
        candidates =  self._reduce_candidates_with_lexicon(candidates, lexicon)
        text = self._select_word(candidates)
        return text, candidates

    def recursion(self, word_img, cuts, currentCut, lexicon, wordString, probability, stateProbabilities, transProbabilities, classes, hypotheses):

        # no cuts possible, return word, probability
        if currentCut == len(cuts)-1:
            if wordString in hypotheses:
                hypotheses[wordString] = max(hypotheses[wordString], probability)
            else:
                hypotheses[wordString] = probability
                logging.info("Found: %s, with prob: %s" % (wordString, probability))
            return

        # Find the next cut
        start = currentCut
        for end in range(start+1, len(cuts)):
            logging.info("Cuts: %s\nStart: %s, End: %s" % (cuts, start, end))
            if not 10 <= (cuts[end] - cuts[start]) < 80:
                logging.info("windows not the right size, %s %s " % (cuts[start],cuts[end]))
                continue
            window = word_img[:,cuts[start]:cuts[end]]
            window = cut_letters.removeWhitelines(window)
            if not window is None:
                f = hog.hog_xeryus(window).reshape(1, -1)
                predictions = self.getPredictions(self.knn, f, classes)
                logging.info("predictions: %s" % predictions)
                # predictions, probabilities = self.knn.predict(f)
                for prediction, prob in predictions.iteritems():

                    # Add the predicted character to the word
                    wordString = wordString + prediction
                    # Add the prediction probability to the total probability
                    probability *= prob
                    # Add the state (position of character) probability to the total probability
                    if prediction in stateProbabilities[len(wordString)-1]:
                        probability *= stateProbabilities[len(wordString)-1][prediction]
                    else:
                        probability *= 0.00001
                    # Add the transition probability to the total probability
                    if len(wordString) > 1:
                        if wordString[-2] in transProbabilities:
                            if wordString[-1] in transProbabilities[wordString[-2]]:
                                probability *= transProbabilities[wordString[-2]][wordString[-1]]
                            else:
                                probability *= 0.00001
                        else:
                            probability *= 0.00001
                    # Set the correct cut index
                    currentCut = end
                    # Into recursion, and beyond!
                    # logging.info("Into recursion with %s, prob: %s" % (wordString, probability))
                    self.recursion(word_img, cuts, currentCut, lexicon, wordString, probability, stateProbabilities, transProbabilities, classes, hypotheses)

    def recursiveRecognize(self, word_img, cuts, lexicon, stateProbabilities, transProbabilities, classes):
        # A graph that keeps track of the possible words, and their probabilities
        hypotheses = {}
        self.recursion(word_img, cuts, cuts[0],    lexicon, "",         1.0,         stateProbabilities, transProbabilities, classes, hypotheses)
        # self.recursion(word_img, cuts, currentCut, lexicon, wordString, probability, stateProbabilities, transProbabilities, hypothese)
        return hypotheses

    def _hypotheses_graph_to_candidates(self, hypotheses):
        possible = [("", 0)]
        for start in sorted(hypotheses.keys()):
            new = []
            for p in possible:
                if start < p[1]:
                    # skip if letters would overlap with the rest
                    # of the word
                    continue
                for l in hypotheses[start]:
                    new.append((p[0] + l[0], l[1]))
            possible = set(possible)
            possible.update(new)
            possible = list(possible)
        # remove duplicates
        possible = sorted(list(set([p[0] for p in possible])))
        return possible

    def _reduce_candidates_with_lexicon(self, candidates, lexicon):
        newCandidates = {}
        for candidate in candidates:
            if candidate in lexicon:
                newCandidates[candidate] = lexicon[candidate]

        return newCandidates

    def _select_word(self, candidates):
        maxVal = 0
        for key, value in candidates.iteritems():
            if value > maxVal:
                word = key
        return word

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
