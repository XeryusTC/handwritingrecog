import logging
import modules.toolbox.croplib as croplib
import modules.toolbox.pamImage as pamImage
import xml.etree.ElementTree as ET

def recognize(preIm, e):
    # Parse the given words (xml) file to find the word sections
    logging.info("Recognizing")
    for sentence in e:
        left = int(sentence.get('left'))
        top = int(sentence.get('top'))
        right = int(sentence.get('right'))
        bottom = int(sentence.get('bottom'))
        cropped_im = croplib.crop(preIm, left, top, right, bottom)
        cropped_im.thisown = True                 # to make Python cleanup the new C++ object afterwards
        cropped_im.save('tmp/sentence'+sentence.get('no')+'.ppm')

        # Now to the individual words
        for word in sentence:
            left = int(word.get('left'))
            top = int(word.get('top'))
            right = int(word.get('right'))
            bottom = int(word.get('bottom'))
            cropped_im = croplib.crop(preIm, left, top, right, bottom)
            cropped_im.thisown = True                 # to make Python cleanup the new C++ object afterwards
            cropped_im.save('tmp/sentence'+sentence.get('no')+'_word'+word.get('no')+'.ppm')

            # classify the word
            word.set('text', 'dunno')

    # Return the new filled in tree
    tree = ET.ElementTree(e)
    logging.info("Recognizing completed")
    return tree
    # shutil.rmtree('tmp')
