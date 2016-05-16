import sys, os, inspect, shutil
import preprocessor as prep
import toolbox.pamImage as pamImage
import toolbox.croplib as croplib
import xml.etree.ElementTree as ET
import cv2

# check commandline parameters
if len(sys.argv) != 4:
    print "Usage: python %s image.ppm input.words /path/to/output.words" % sys.argv[0]
    sys.exit(1)

# Create a temp folder for file storage
if not os.path.exists("tmp"):
    os.makedirs("tmp")

img = cv2.imread(sys.argv[1], 0)
words_file_name = sys.argv[2]
words_file_name_out = sys.argv[3]

# Pre-proccesing steps

if prep.preprocess(img) != 0:
    print "Something went wrong in Pre-proccesing"
    sys.exit(1)
else :
    print "Pre-processing completed"

# Get the pre-processed image
preIm = pamImage.PamImage("tmp/preprocessed.ppm")
e = ET.parse(words_file_name).getroot()

print "Parsing pre-processed image into individual sentences"
# Parse the given words (xml) file to find the word sections
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
        print "Set text: %s" % word.get('text')

# Write results/delete the tmp folder
tree = ET.ElementTree(e)
tree.write(words_file_name_out)
# shutil.rmtree('tmp')
