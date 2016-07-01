import sys, os, inspect, shutil, fnmatch, logging
from unipath import Path, DIRS_NO_LINKS
import recognizer
import numpy as np
import cv2
import csv

def param_sweep():
    test_dir = Path('test_files/words/KNMP')
    file_range = np.arange(4,50,2)
    # file_range = [10]

    minCutWindow_range = np.arange(0,40,5)
    minCutWindow_range[0] = 1
    # minCutWindow_range = [1]

    maxCutWindow_range = np.arange(60,210,10)
    # maxCutWindow_range = [60]

    globalLexicon_range = [True, False]
    # globalLexicon_range = [True]

    with open('results.csv', 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')

        results = []
        labels = ["minWindow", "maxWindow", "GlobalLexicon", "Right", "Wrong", "Accuracy"]
        results.append(labels)
        csv_writer.writerow(labels)

        for minCutWindow in minCutWindow_range:
            for maxCutWindow in maxCutWindow_range:
                for globalLexicon in globalLexicon_range:
                    logging.info("Now testing %s\t%s\t%s" % (minCutWindow, maxCutWindow, globalLexicon))
                    globalRight = 0.0
                    globalWrong = 0.0
                    for file_nr in file_range:
                        logging.info("File: %s" % str(file_nr))

                        img_file = str(file_nr) + ".jpg"
                        img_path = Path(test_dir, img_file)
                        logging.info("img path %s" % img_path)
                        img = cv2.imread(img_path, 0)

                        words_file = str(file_nr) + ".words"
                        words_file_name = Path(test_dir, words_file)
                        right, wrong = recognizer.main(
                            state="internal", img=img, words_file_name=words_file_name,
                            minCutWindow = minCutWindow, maxCutWindow=maxCutWindow,
                            globalLexicon=globalLexicon
                        )
                        globalRight += right
                        globalWrong += wrong
                    # Log the results
                    accuracy = globalRight / (globalRight + globalWrong)
                    result = [minCutWindow, maxCutWindow, globalLexicon, globalRight, globalWrong, accuracy]
                    logging.info("Interim result:")
                    logging.info(result)
                    results.append(result)
                    csv_writer.writerow(result)
        logging.info("Total result")
        logging.info(results)

if __name__ == '__main__':
    param_sweep()
