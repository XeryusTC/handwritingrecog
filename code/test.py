import sys
import cv2
in_file = sys.argv[1]
img = cv2.imread(in_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img, contours, hierarchy = cv2.findContours(100, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
