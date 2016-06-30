import numpy as np
import pickle

if __name__ == '__main__':
    letters = {}
    for dataset in ['KNMP', 'Stanford']:
        with open('window_stats_' + dataset + '.csv', 'r') as f:
            f.readline() # read header
            for line in f:
                values = line.split(',')
                if values[0] in letters:
                    letters[values[0]].append(int(values[1]))
                else:
                    letters[values[0]] = [int(values[1])]
    stats = {}
    for letter in letters:
        stats[letter] = [np.mean(np.asarray(letters[letter])), np.std(np.asarray(letters[letter]))]

    pickle.dump(stats, open("window_means.pickle", "wb"))
