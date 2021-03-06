import sys, os, logging, shutil
import xml.etree.ElementTree as ET
from unipath import Path
import cv2
import pickle
import logging
import numpy as np
import operator
import Levenshtein as Lev

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
        #with open('tmp/svm-pretrained.pickle', 'r') as f:
        #    self.svm = pickle.load(f)
        with open('tmp/knn.pickle', 'r') as f:
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
            print "image not good for classifying"
            return None

    def getPredictions(self, clf, featureVector, classes):
        predictions = {}
        probs = clf.predict_proba(featureVector)
        for idx, val in enumerate(probs[0]):
            if val > 0.3 or val == max(probs[0]):
                predictions[classes[idx]] = val
                #  print 'prob: %s, class: %s' % (val, classes[idx])
        return predictions

    def lexiconLevenshtein(self, hypotheses, lexicon):
        loopWords = lexicon
        candidates = []
        while True:
            if hypotheses:
                bestString = max(hypotheses.iteritems(), key=operator.itemgetter(1))[0]
                hypotheses.pop(bestString, None)
                minDistance = 100
                for word in loopWords:
                    distance = Lev.distance(word, bestString)
                    if distance < minDistance:
                        candidates = [word]
                        minDistance = distance
                    elif distance == minDistance:
                        candidates.append(word)
                if len(candidates) == 1:
                    break
                if len(candidates) == 0:
                    candidates = loopWords
                    break
                loopWords = candidates
            else:
                break
        return candidates

    def recursion(self, word_img, cuts, currentCut, lexicon, wordString, probability, stateProbabilities, transProbabilities, classes, hypotheses, savedPredictions, minCutWindow, maxCutWindow):
        if currentCut == len(cuts)-1:
            if wordString in hypotheses:
                hypotheses[wordString] = max(hypotheses[wordString], probability)
            else:
                hypotheses[wordString] = probability
            return

        # Find the next cut
        start = currentCut
        for end in range(start+1, len(cuts)):
            if not minCutWindow <= (cuts[end] - cuts[start]) < maxCutWindow:
                continue
            window = word_img[:,cuts[start]:cuts[end]]
            window = cut_letters.removeWhitelines(window)

            if not window is None:
                f = hog.hog_xeryus(window).reshape(1, -1)

                # First check if the selected combination has been predicted before
                if str(start) in savedPredictions and str(end) in savedPredictions[str(start)]:
                    predictions = savedPredictions[str(start)][str(end)]
                else: # Else calculate it and add to database
                    predictions = self.getPredictions(self.knn, f, classes)
                    savedPredictions[str(start)][str(end)] = predictions

                for prediction, prob in predictions.iteritems():
                    # Add the predicted character to the word
                    wordString = wordString + prediction
                    # Add a factor to overcome advantages for large windows
                    factor = end-start
                    # Add the prediction probability to the total probability
                    probability += factor * np.log10(prob)

                    # Add the state (position of character) probability to the total probability
                    if len(wordString) <= len(stateProbabilities):
                        if prediction in stateProbabilities[len(wordString)-1]:
                            probability += factor * stateProbabilities[len(wordString)-1][prediction]
                        else:
                            continue
                            # probability -= 5
                    else:
                        continue
                        #probability -= 5

                    # Add the transition probability to the total probability
                    if len(wordString) > 1:
                        if wordString[-2] in transProbabilities:
                            if wordString[-1] in transProbabilities[wordString[-2]]:
                                probability += factor * transProbabilities[wordString[-2]][wordString[-1]]
                            else:
                                continue
                                # probability -= 5
                        else:
                            continue
                            # probability -= 5

                    # Set the correct cut index
                    currentCut = end
                    # Into recursion, and beyond!
                    self.recursion(
                        word_img, cuts, currentCut, lexicon, wordString,
                        probability, stateProbabilities, transProbabilities,
                        classes, hypotheses, savedPredictions, minCutWindow,
                        maxCutWindow
                    )

    def recursiveRecognize(
            self, word_img, cuts, lexicon, stateProbabilities,
            transProbabilities, classes, minCutWindow, maxCutWindow
    ):
        # A graph that keeps track of the possible words, and their probabilities
        hypotheses = {}
        # Save hypotheses per cut combination
        savedPredictions = {}
        for x in range(len(cuts)):
            savedPredictions[str(x)] = {}
        self.recursion(
            word_img, cuts, cuts[0], lexicon, "", 0.0, stateProbabilities,
            transProbabilities, classes, hypotheses, savedPredictions,
            minCutWindow, maxCutWindow
        )
        hypotheses = self.lexiconLevenshtein(hypotheses, lexicon)

        # If Levenshtein returns one hypothesis return it, else return the most probable one
        if len(hypotheses) > 1:
            maxCount = 0
            for word in hypotheses:
                if lexicon[word] > maxCount:
                    estimate = word
                    maxCount = lexicon[word]
        elif len(hypotheses) == 0:
            estimate = max(lexicon.iteritems(), key=operator.itemgetter(1))[0]
        else:
            estimate = hypotheses[0]

        return estimate

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
