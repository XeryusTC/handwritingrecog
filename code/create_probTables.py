import pickle

def create_stateProbs(lexicon):
    longestWord = len(max(lexicon, key=len))
    pi = [{} for _ in xrange(longestWord)]

    for index, probs in enumerate(pi):
        total = 0.0
        # Count letter occurences on this index
        for key in lexicon:
            if index < len(key):
                total += lexicon[key]
                if key[index] in probs:
                    probs[key[index]] += lexicon[key]
                else:
                    probs[key[index]] = lexicon[key]
        # Calculate state probabilities
        for letter in probs:
            probs[letter] = float(probs[letter]) / total
    return pi

def create_transProbs(lexicon):
    T = {}
    total = 0.0

    # Count transition occurences
    for word in lexicon:
        for index, letter in enumerate(word):
            if index + 1 < len(word):
                total += lexicon[word]
                if letter in T:
                    if word[index + 1] in T[letter]:
                        T[letter][word[index+1]] += lexicon[word]
                    else:
                        T[letter][word[index+1]] = lexicon[word]
                else:
                    T[letter] = {}
                    T[letter][word[index + 1]] = lexicon[word]
    # Calculate transition probabilities
    for letter in T:
        for letter2 in T[letter]:
            T[letter][letter2] = float(T[letter][letter2]) / total
    return T

if __name__ == '__main__':
    lex = {}
    with open("tmp/lexicon.csv") as f:
        for line in f:
            (key, val) = line.split(',')
            lex[key] = int(val)

    pi = create_stateProbs(lex)
    pickle.dump(pi, open("stateProbs.pickle", "wb"))
    T = create_transProbs(lex)
    pickle.dump(T, open("transProbs.pickle", "wb"))

    print T
