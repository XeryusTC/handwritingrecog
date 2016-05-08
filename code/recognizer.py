import sys, os, inspect, shutil
import own_modules.preprocess as prep
import toolbox.pamImage as pamImage
import toolbox.croplib as croplib
import xml.etree.ElementTree
# check commandline parameters
if len(sys.argv) != 4:
    print "Usage: python %s image.ppm input.words /path/to/output.words" % sys.argv[0]
    sys.exit(1)

# Create a temp folder for file storage
if not os.path.exists("tmp"):
    os.makedirs("tmp")

in_file_name = sys.argv[1]
words_file_name = sys.argv[2]

# Pre-proccesing steps

if prep.preprocess(in_file_name) != 0:
    print "Something went wrong in Pre-proccesing"
    sys.exit(1)
else :
    print "Pre-processing completed"

preIm = pamImage.PamImage("tmp/preprocessed.ppm")
e = xml.etree.ElementTree.parse(words_file_name).getroot()

print "Parsing pre-processed image into individual sentences"
# Parse the given words (xml) file to find the word sections
for child in e:
    left = int(child.get('left'))
    top = int(child.get('top'))
    right = int(child.get('right'))
    bottom = int(child.get('bottom'))
    cropped_im = croplib.crop(preIm, left, top, right, bottom)
    cropped_im.thisown = True                 # to make Python cleanup the new C++ object afterwards
    cropped_im.save('tmp/sentence'+child.get('no')+'.ppm')

# Write results/delete the tmp folder
# shutil.rmtree('tmp')
