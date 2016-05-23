# -*- coding: utf-8 -*-
import sys, os, cv2, glob, csv, shutil
import numpy as np
import logging
# from scipy.cluster.vq import whiten, kmeans2
# from scipy.spatial.distance import cdist, pdist
from unipath import Path, DIRS_NO_LINKS

### Old main function with kmeans also
'''
def main_xeryus():
    work_dir = Path("tmp/segments")
    if not work_dir.exists():
        print "You must first run create_labels.py"
        sys.exit(1)

    data = []
    labels = work_dir.listdir(filter=DIRS_NO_LINKS)
    for label in labels:
        images = label.listdir(pattern='*.ppm')
        print label.stem, "size:", len(images)
        for image in images:
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            f = hog_xeryus(img)
            data.append(f.flatten())

    data = whiten(data)
    centroids, labels = kmeans2(data, len(labels), minit='points')

    cluster_sizes = {}
    for i in labels:
        if i not in cluster_sizes:
            cluster_sizes[i] = 0;
        cluster_sizes[i] += 1
    for cluster in cluster_sizes:
        print "Cluster", cluster, "size:", cluster_sizes[cluster]

    # Calculate some metrics
    D = cdist(data, centroids, 'euclidean')
    cIdx = np.argmin(D, axis=1)
    dist = np.min(D, axis=1)

    tot_withinss = sum(dist**2)
    totss = sum(pdist(data)**2) / data.shape[0]
    betweenss = totss - tot_withinss
    print tot_withinss, totss, betweenss

'''

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

def hog_alternative(img):
    img = cv2.resize(img,(64, 128), interpolation = cv2.INTER_CUBIC)
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def doHog(imgDir, hogDir, hog = "xeryus"):
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
            img = cv2.imread(os.path.join(subdir, f))
            if hog == "xeryus":
                hist = hog_xeryus(img)
            else:
                hist = hog_alternative(img)

            features.append(hist[:,0])
            labels.append(os.path.basename(os.path.normpath(subdir)))

    np.save(hogDir + 'hog', features)
    np.save(hogDir + 'labels', labels)
