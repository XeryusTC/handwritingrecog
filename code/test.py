import sys
import cv2

im = cv2.imread('Stanford-CCCC_0002.ppm')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
#ret, result =  cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#result, contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#img, contours, hierarchy = cv2.findContours(100, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
