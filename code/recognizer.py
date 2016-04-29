import sys, os, inspect
# import C++ libraries
# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"toolbox")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

print sys.path
import toolbox.pamImage as pamImage, toobox.croplib as croplib

# check commandline parameters
if len(sys.argv) != 4:
    print "Example program. Crops the image to a smaller region."
    print "Usage: python %s image.ppm input.words /path/to/output.words" % sys.argv[0]
    sys.exit(1)

in_file_name = sys.argv[1]
out_file_name = 'output' + in_file_name

# open image
im = pamImage.PamImage(in_file_name)
width, height = im.getWidth(), im.getHeight()
print "Image width:", width
print "Image height:", height
