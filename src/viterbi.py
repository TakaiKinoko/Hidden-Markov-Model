#! /usr/bin/python

__author__="Fang Han Cabrera <fh643@nyu.edu>"
__date__ ="$Dec 3, 2019"

'''
THE VITERBI ALGORITHM WITH BACKPOINTERS
Input:  a sentence x1, ... xn,
        parameter q(s|u, v)
        parameter e(x|s)

Initialization Set pi(0, *, *) = 1

Definition: S_-1 = S_0 = {*}, S_k = S for k in [1, n]

Algorithm:
    for k = 1 ... n,
        for u in S_(k-1), v in S_k
            pi(k, u, v) = max (pi(k-1, w, u) × q(v| w, u) × e(x_k|v)) for w in S_(k-2)
            bp(k, u, v) = argmax (pi(k-1, w, u) × q(v| w, u) × e(x_k|v)) for  w in S_(k-2)

    Set (y_(n−1), yn) = argmax(pi(n, u, v) × q(STOP|u, v)) for (u,v)

    for k = (n-1) ... 1, y_k = bp(k+2, y_(k+1), y_(k+2))

    return the tag sequence y1 ... yn
'''

from collections import defaultdict
import collections
import math

def get_counts(counts_file):
    '''
    :param counts_file: file containing {unigram, bigram, trigram, emission} counts,
    in the form of <count> <WORDTAG|1-GRAM|2-GRAM|3-GRAM> <tag> <word>
    TODO change description
    a pair of dictionaries where the first is emission counts dict, the second ngram count dict.

     ngram count up to tri-grams
    '''
    # tags
    NE = set()
    # emission counts: Count(y, x). key: y  value: dict mapping word (x) to Count(y, x)
    emission_cnts = collections.defaultdict(dict)  # use defaultdict for nested dict
    # ngram dictionary: Count(y), Count(yn-1, yn), Count(yn-2, yn-1, yn)
    ngram_cnts = collections.defaultdict(int)

    # read count file and count words and n-grams
    for line in counts_file.read().splitlines():
        parts = line.strip().split(' ')
        count = int(parts[0])
        cat = parts[1]  # category

        if cat == 'WORDTAG':  # emission counts
            emission_cnts[parts[2]][parts[3]] = count
            ngram_cnts[parts[3]] += count
            NE.add(parts[2])
        if cat in {'1-GRAM', '2-GRAM', '3-GRAM'}:  # N-GRAM counts
            ngram_cnts[tuple(
                parts[2:])] = count  # using tuple as key will make it easier to query the dict with unkown length list
    return (NE, emission_cnts, ngram_cnts)

def compute_q(ngram_cnts):
    '''
    r(y_-1, y_0, y_1, ..., y_k) = product(q(y_i|y_i-2, y_i-1)) * product(e(x_i|y_i)) for all i in [1, k]

    This function computes the q(y_i|y_i-2, y_i-1) and stores them in a dictionary
    '''
    Q = collections.defaultdict(float)
    for k, v in ngram_cnts.items():
        if type(k) == tuple and len(k) == 3:
            Q[k] = float(v) / float(ngram_cnts[k[:-1]])
    return Q

def compute_e(emission_cnts):
    '''
    r(y_-1, y_0, y_1, ..., y_k) = product(q(y_i|y_i-2, y_i-1)) * product(e(x_i|y_i)) for all i in [1, k]

    This function computes the e(x_i|y_i) and stores them in a dictionary
    '''
    E = collections.defaultdict(float)  # map (word, pos) pairs to its emission parameter

    for pos in emission_cnts.keys():
        count_y = sum(emission_cnts[pos].values())
        for x in emission_cnts[pos].keys():
            E[(x, pos)] = float(emission_cnts[pos][x]) / float(count_y)
    return E

def read_sentences(file):
    '''
    :param file: file containing test data (format: each word takes a line, an empty line separating different sentences)

    produce a list of sentences where unkoown words are replaced as "_RARE_"
    '''
    # TODO delete
    #file = open("../data/ner_dev.dat", "r")
    sentences = []
    tmp = []
    for line in file.read().splitlines():
        parts = line.strip().split(' ')
        w = parts[0]
        if w == '':
            sentences.append(tmp)
            tmp = []
        else:
            tmp.append(w)
    #print(self.sentences)
    return sentences

def tag_rares(sentences, ngram_cnts):
    '''
    replace unseem words in the sentences with '_RARE_'
    :param sentences: list of lists, where each nested list is a sentence
    :param ngram_cnts: a list mapping each ngram and wordtag to its frequency
    :return: list of the pairs of sentences where the first is tagged with rare, the second original
    '''
    sen_with_rare = []
    for s in sentences:
        s_new = []
        # unseen replace with '_RARE_'
        for word in s:
            if ngram_cnts[word] == 0:
                s_new.append("_RARE_")
            else:
                s_new.append(word)
        sen_with_rare.append((s_new, s))
    return sen_with_rare

def construct_dp_table(sentence, transmission, emission, NE_set):
    '''
    define a dynamic programming table (see columnbia slide about HMM)

    pi(k, u, v) = maximum probability of a tag sequence ending in tags u, v at position k

    that is:
    pi(k, u, v) = max(r(y_-1, y_0, y_1, ... y_k)) where y_(k-1) = u, y_k = v

    base case:
    pi(0, *, *) = 1
    '''

    PI = defaultdict(int)
    # backpointers
    BP = defaultdict(str)

    # base case
    PI[(0, "*", "*")] = 1

    # Dynamic programming
    # for k = 1 ... n,
    for k in range(1, len(sentence) + 1):
        # for v in S_k
        for v in NE_set:
            # for u in S_(k-1)
            for u in NE_set if k > 1 else {'*'}:
                # for w in S_(k-2)
                for w in NE_set if k > 2 else {'*'}:
                    k_prob = PI[(k - 1, w, u)] * transmission[(w, u, v)] * emission[(sentence[k - 1]), v]
                    if k_prob > PI[(k, u, v)]:
                        PI[(k, u, v)] = k_prob
                        BP[(k, u, v)] = w

    return PI, BP

def get_tags_and_probs(BP, NE, PI, Q, sentence):
    # resulting tags
    tags = []
    # resulting log probabilities
    log_probs = []

    n_prob = 0
    y_n = ''
    y_n_1 = ''

    # get yn y(n-1)
    # Set (yn−1, yn) = arg max(u,v) (PI(n, u, v) × Q(STOP|u, v))
    for u in NE if len(sentence) > 1 else {'*'}:
        for v in NE:
            if PI[(len(sentence), u, v)] * Q[(u, v, "STOP")] > n_prob:
                n_prob = PI[(len(sentence), u, v)] * Q[(u, v, "STOP")]
                y_n = u
                y_n_1 = v
    v = y_n_1
    tags.append(v)
    u = y_n
    tags.append(u)
    p_n_1 = math.log(n_prob, 2)
    log_probs.append(p_n_1)
    p_n = math.log(PI[(len(sentence), u, v)], 2)
    log_probs.append(p_n)

    # get all ys in tags through BP, backwards of course
    for k in range(len(sentence), 1, -1):
        w = BP[(k, u, v)]
        tags.append(w)
        log_p = math.log(PI[(k - 1, w, u)], 2)
        log_probs.append(log_p)
        v = u
        u = w

    tags.reverse()
    log_probs.reverse()

    return tags, log_probs

def viterbi_tagger(counts, ner_dev, output):

    NE, emission_cnts, ngram_cnts = get_counts(counts)
    # transmission
    Q = compute_q(ngram_cnts)
    # emission
    E = compute_e(emission_cnts)
    # get the list of all sentences, and the list of words in a sentence.
    sentences = read_sentences(ner_dev)
    # replace unseen words in the test set with '_RARE_'
    sens_with_rare = tag_rares(sentences, ngram_cnts)

    # iterate over sentences
    for s, original in sens_with_rare:
        #  PI: maximum probability of a tag sequence ending in tags u, v at position k, BP: backpointers
        PI, BP = construct_dp_table(s, Q, E, NE)

        tags, log_probs = get_tags_and_probs(BP, NE, PI, Q, s)

        # output the tags for the sentence
        for count, word in enumerate(original):
            output.write(str(word) + ' ' + str(tags[count + 1]) + ' ' + str(log_probs[count]) + '\n')
        output.write('\n')

if __name__ == "__main__":
    with open("../output/ner_with_rare.counts", "r") as counts:
        with open("../input/ner_dev.dat", "r") as ner_dev:
            with open("../output/sen_tagged.predict", "w+") as output:
                viterbi_tagger(counts, ner_dev, output)
    output.close()
    ner_dev.close()
    counts.close()
