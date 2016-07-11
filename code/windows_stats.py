from __future__ import print_function
from unipath import Path
import numpy as np

if __name__ == '__main__':
    workdir = Path('tmp/')
    datasets = workdir.listdir('window_stats_*.csv')
    total_widths = []
    total_heights = []

    for data in datasets:
        with open(data, 'r') as f:
            widths = []
            heights = []
            for line in f.readlines():
                letter, width, height = line.strip().split(',')
                if letter=='label' and width=='width' and height=='height':
                    continue # skip header
                widths.append(int(width))
                heights.append(int(height))
                total_widths.append(int(width))
                total_heights.append(int(height))

            print(str(data), 'statistics:')
            print('Mean  width: ', np.mean(widths))
            print('Stdev width: ', np.std(widths))
            print('Mean  height:', np.mean(heights))
            print('Stdev height:', np.std(heights))
            print('Median width:', np.median(widths))
            print('Median height:', np.median(heights))
            print()

    print('Total statistics:')
    print('Mean  width: ', np.mean(total_widths))
    print('Stdev width: ', np.std(total_widths))
    print('Mean  height:', np.mean(total_heights))
    print('Stdev height:', np.std(total_heights))
    print('Median width:', np.median(total_widths))
    print('Median height:', np.median(total_heights))
