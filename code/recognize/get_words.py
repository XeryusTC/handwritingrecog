import sys, os, logging, shutil
import xml.etree.ElementTree as ET

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import toolbox.croplib as croplib
import toolbox.pamImage as pamImage

def getWords(preIm, xml, sentenceDir, wordDir):
    if os.path.exists(sentenceDir):
        shutil.rmtree(sentenceDir)
    os.makedirs(sentenceDir)
    if os.path.exists(wordDir):
        shutil.rmtree(wordDir)
    os.makedirs(wordDir)

    # Parse the given words (xml) file to find the word sections
    logging.info("Cropping words")
    for sentence in xml:
        left = int(sentence.get('left'))
        top = int(sentence.get('top'))
        right = int(sentence.get('right'))
        bottom = int(sentence.get('bottom'))
        cropped_im = croplib.crop(preIm, left, top, right, bottom)
        cropped_im.thisown = True                 # to make Python cleanup the new C++ object afterwards
        cropped_im.save(sentenceDir + 'sentence'+ sentence.get('no')+'.ppm')

        # Now to the individual words
        for word in sentence:
            left = int(word.get('left'))
            top = int(word.get('top'))
            right = int(word.get('right'))
            bottom = int(word.get('bottom'))
            cropped_im = croplib.crop(preIm, left, top, right, bottom)
            cropped_im.thisown = True                 # to make Python cleanup the new C++ object afterwards
            cropped_im.save(wordDir + sentence.get('no') + '_word' + word.get('no')+'.ppm')

            # classify the word
            # word.set('text', 'dunno')

    # Return the new filled in tree
    tree = ET.ElementTree(xml)
    logging.info("Words cropped")
    return tree
    # shutil.rmtree('tmp')
