# Lets do some processing
import sys
import pamImage, croplib

# check commandline parameters
if len(sys.argv) != 3:
    print "Example program. Crops the image to a smaller region."
    print "Usage: python %s image.ppm outfile.ppm" % sys.argv[0]
    sys.exit(1)

in_file_name = sys.argv[1]
out_file_name = sys.argv[2]

# open image
im = pamImage.PamImage(in_file_name)
width, height = im.getWidth(), im.getHeight()
print "Image width:", width
print "Image height:", height

print "This image is a ", im.getImageType(), " type"
print "Ascii-art something"
im.printAsciiArt()
