# Packages
import sys, os, cv2, uuid, logging
import logging.config
import numpy as np
from unipath import Path, DIRS_NO_LINKS
from scipy.cluster.vq import whiten, kmeans2
import xml.etree.ElementTree as ET

# Own modules
from general.preprocessor import preprocess

logging.config.fileConfig('logging.conf')

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['KNMP', 'Stanford']:
        print("Usage: python %s <dataset>" % sys.argv[0])
        print("\tDataset should be either 'KNMP' or 'Stanford'")
        sys.exit(1)
    create(sys.argv[1])

def create(dataset):
    # create a clean temporary directory
    work_dir = Path("tmp")
    work_dir.rmtree()
    work_dir.mkdir()

    # Find all the pages in the dataset
    img_dir = Path(Path.cwd().ancestor(1), 'data/hwr_data/pages', dataset)
    ann_dir = Path(Path.cwd().ancestor(1), 'data/charannotations')
    images = img_dir.listdir('*.jpg')
    annotations = ann_dir.listdir(dataset + '*.words')
    files = merge(images, annotations)

    # Create character segmentations
    stats = {}
    for f in files:
        # Preprocess
        logging.info("Preprocessing %s", str(f[0]))
        pagesPathFolder = Path(work_dir, 'pages')
        pagesPathFolder.mkdir()
        pagePath = Path(pagesPathFolder, f[0].stem + '.ppm')
        img = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE)
        img = preprocess(img)
        cv2.imwrite(pagePath, img)

        # Segment
        segmentPathFolder = Path(work_dir, 'segments')
        segmentPathFolder.mkdir()
        e = ET.parse(f[1]).getroot()
        logging.info("Segmenting %s", str(f[0]))
        segment(img, e, segmentPathFolder, stats)

    print_statistics(stats)

def merge(images, annotations):
    ret = []
    for img in images:
        for ann in annotations:
            if img.stem == ann.stem:
                ret.append((img, ann))
    return ret

def segment(img, annotation, work_dir, stats):
    sides = ['left', 'top', 'right', 'bottom']
    # Parse the given sentences
    for sentence in annotation:
        for word in sentence:
            for char in word:
                c = char.get('text')
                if c in '!-,.':
                    continue # Skip stupid labels
                cdir = Path(work_dir, c)
                cdir.mkdir()
                f = Path(cdir, str(uuid.uuid1()) + '.ppm')

                rect = {side: int(char.get(side)) for side in sides}
                # Correct for swapped coordinates
                if rect['top'] > rect['bottom']:
                    rect['top'], rect['bottom'] = rect['bottom'], rect['top']
                if rect['left'] > rect['right']:
                    rect['left'], rect['right'] = rect['right'], rect['left']
                cropped_im = img[rect['top']:rect['bottom'], rect['left']:rect['right']]

                # Remove rows from the top if they're white
                while cropped_im.shape[1] > 0 and cropped_im.shape[0] > 0 \
                        and min(cropped_im[0,:]) == 255:
                    cropped_im = cropped_im[1:,:]
                # Remove from the bottom
                while cropped_im.shape[1] > 0 and cropped_im.shape[0] > 0 \
                        and min(cropped_im[-1,:]) == 255:
                    cropped_im = cropped_im[:-1,:]
                # Remove from the left
                while cropped_im.shape[1] > 0 and cropped_im.shape[0] > 0 \
                        and min(cropped_im[:,0]) == 255:
                    cropped_im = cropped_im[:,1:]
                # Remove from the right
                while cropped_im.shape[1] > 0 and cropped_im.shape[0] > 0 \
                        and min(cropped_im[:,-1]) == 255:
                    cropped_im = cropped_im[:,:-1]
                if cropped_im.shape[0] <= 5 or cropped_im.shape[1] <= 5:
                    print "Discarding image"
                    continue
                cv2.imwrite(f, cropped_im)

                # Add to statistics
                if c not in stats.keys():
                    stats[c] = {'width': [], 'height': []}
                stats[c]['width'].append(cropped_im.shape[1])
                stats[c]['height'].append(cropped_im.shape[0])

def print_statistics(stats):
    min_width   = {}
    min_height  = {}
    max_width   = {}
    max_height  = {}
    mean_width  = {}
    mean_height = {}
    med_width   = {}
    med_height  = {}
    all_width   = []
    all_height  = []
    with open('window_stats.csv', 'w') as f:
        f.write('label,width,height')
        for c, v in stats.items():
            for i in range(len(v['width'])):
                f.write('{},{},{}\n'.format(c, v['width'][i], v['height'][i]))
            min_width[c]   = min(v['width'])
            max_width[c]   = max(v['width'])
            mean_width[c]  = sum(v['width']) / len(v['width'])
            med_width[c]   = sorted(v['width'])[len(v['width']) / 2]
            min_height[c]  = min(v['height'])
            max_height[c]   = max(v['height'])
            mean_height[c] = sum(v['height']) / len(v['height'])
            med_height[c]  = sorted(v['height'])[len(v['height']) / 2]
            all_width.extend(v['width'])
            all_height.extend(v['height'])

            print "Statistics for label '{}':".format(c)
            print "\tmin:    {:>4}\t{:>4}".format(min_width[c], min_height[c])
            print "\tmax:    {:>4}\t{:>4}".format(max_width[c], max_height[c])
            print "\tmean:   {:>4}\t{:>4}".format(mean_width[c], mean_height[c])
            print "\tmedian: {:>4}\t{:>4}".format(med_width[c], med_height[c])


    print "Summary of statistics:"
    print "\tmin:    {:>4}\t{:>4}".format(min(min_width.values()),
        min(min_height.values()))
    print "\tmax:    {:>4}\t{:>4}".format(max(max_width.values()),
        max(max_width.values()))
    meanw = sum(mean_width.values()) / len(mean_width.values())
    meanh = sum(mean_height.values()) / len(mean_height.values())
    print "\tmean:   {:>4}\t{:>4}".format(meanw, meanh)
    medw = sorted(all_width)[len(all_width) / 2]
    medh = sorted(all_height)[len(all_height) / 2]
    print "\tmedian: {:>4}\t{:>4}".format(medw, medh)
    print "Total segments:", len(all_width)

if __name__ == '__main__':
    main()
