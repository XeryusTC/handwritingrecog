# -*- coding: utf-8 -*-
import sys, os, cv2, glob, csv, shutil
import numpy as np
import logging
# from scipy.cluster.vq import whiten, kmeans2
# from scipy.spatial.distance import cdist, pdist
from unipath import Path, DIRS_NO_LINKS

def hog_xeryus(img, char_size=(72, 72), window_size=(80, 80), block_size=(2, 2),
        cell_size=(8, 8), nbins=9):
    scale = max(img.shape[0] / float(char_size[0]),
        img.shape[1] / float(char_size[1]))
    # Resize and add padding
    img = cv2.resize(img, char_size)
    border_h = (window_size[0] - char_size[0]) / 2
    border_w = (window_size[1] - char_size[1]) / 2
    img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w,
        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Compute HOG feature
    hog = cv2.HOGDescriptor(_winSize=window_size,
        _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins = nbins)
    f = hog.compute(img)
    return f

def doHog(imgDir, hogDir):
    if not os.path.exists(imgDir):
        print("You must first create segmented images")
        sys.exit(1)

    if os.path.exists(hogDir):
        shutil.rmtree(hogDir)
    os.makedirs(hogDir)

    features = []
    labels = []

    for subdir, dirs, files in os.walk(imgDir):
        # print os.path.basename(os.path.normpath(subdir))
        for f in files:
            # logging.info("Hogging file %s" % f)
            img = cv2.imread(os.path.join(subdir, f))
            hist = hog_xeryus(img)
            #print(hist.shape)
            features.append(hist[:,0])

            labels.append(os.path.basename(os.path.normpath(subdir)))

    np.save(hogDir + 'hog', features)
    np.save(hogDir + 'labels', labels)
