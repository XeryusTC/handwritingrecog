import numpy as np
import pickle

def create():
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
    meanLetter = 0
    stdLetter = 0
    for letter in letters:
        stats[letter] = [np.mean(np.asarray(letters[letter])), np.std(np.asarray(letters[letter]))]
        meanLetter += np.mean(np.asarray(letters[letter]))
        stdLetter += np.std(np.asarray(letters[letter]))

    meanLetter /= len(letters)
    stdLetter /= len(letters)

    lexicon_means_stds = {}
    with open("tmp/lexicon.csv") as l:
        for line in l:
            (key, val) = line.split(',')
            meanWord = 0
            stdWord = 0
            for letter in key:
                if letter in stats:
                    meanWord += stats[letter][0]
                    stdWord += stats[letter][1]
                else:
                    meanWord += meanLetter
                    stdWord += stdLetter
            lexicon_means_stds[key] = [meanWord, stdWord]

    pickle.dump(lexicon_means_stds, open("tmp/lexicon_means_stds.pickle", "wb"))
    return lexicon_means_stds

if __name__ == '__main__':
    create()
