# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
from scipy.cluster.vq import whiten, kmeans2
from scipy.spatial.distance import cdist, pdist
from unipath import Path, DIRS_NO_LINKS

def main():
    work_dir = Path("tmp")
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
            f = hog(img)
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

def hog(img, char_size=(72, 72), window_size=(80, 80), block_size=(2, 2),
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

if __name__ == '__main__':
    main()
