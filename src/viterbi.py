#! /usr/bin/python

__author__="Fang Han Cabrera <fh643@nyu.edu>"
__date__ ="$Dec 3, 2019"

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
    K = set()
    # emission counts: Count(y, x). key: y  value: dict mapping word (x) to Count(y, x)
    emission_cnts = collections.defaultdict(dict)  # use defaultdict for nested dict
    # ngram dictionary: Count(y), Count(yn-1, yn), Count(yn-2, yn-1, yn)
    ngram_cnts = collections.defaultdict(int)
    # word counts

    # read count file and count words and n-grams
    for line in counts_file.read().splitlines():
        parts = line.strip().split(' ')
        count = int(parts[0])
        cat = parts[1]  # category

        if cat == 'WORDTAG':  # emission counts
            emission_cnts[parts[2]][parts[3]] = count
            ngram_cnts[parts[3]] += count
            K.add(parts[2])
        if cat in {'1-GRAM', '2-GRAM', '3-GRAM'}:
            ngram_cnts[tuple(
                parts[2:])] = count  # using tuple as key will make it easier to query the dict with unkown length list
    return (K, emission_cnts, ngram_cnts)

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

    for y in emission_cnts.keys():
        cnt_y = sum(emission_cnts[y].values())
        for x in emission_cnts[y].keys():
            E[(x, y)] = float(emission_cnts[y][x]) / float(cnt_y)
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


def viterbi_tagger(counts, ner_dev, output):

    # word counts
    #emission_cnts = defaultdict(dict)
    # grams counts
    #ngram_cnts = defaultdict(int)
    # e(x|y)
    #e = defaultdict(float)
    # q(u, v, w)
    #q = defaultdict(float)
    # possible tags
    #K = set()

    K, emission_cnts, ngram_cnts = get_counts(counts)

    q = compute_q(ngram_cnts)
    e = compute_e(emission_cnts)

    # get the list of all sentences, and the list of words in a sentence.
    sentences = read_sentences(ner_dev)

    # for each sentence
    for s in sentences:
        s_org = s
        s = []
        # unseen replace with '_RARE_'
        for word in s_org:
            if ngram_cnts[word] == 0:
                s.append("_RARE_")
            else:
                s.append(word)
        # pi(k, u, v)
        pi = defaultdict(int)
        # backpointers
        bp = defaultdict(str)
        pi[(0, "*", "*")] = 1
        n = len(s)

        # dynamically get max pi and keep backpointers
        for k in range(1, n + 1):
            for v in K:
                for u in K if k > 1 else {'*'}:
                    for w in K if k > 2 else {'*'}:
                        k_prob = pi[(k - 1, w, u)] * q[(w, u, v)] * e[(s[k - 1]), v]
                        if k_prob > pi[(k, u, v)]:
                            pi[(k, u, v)] = k_prob
                            bp[(k, u, v)] = w
        # return list of tags Y
        Y = []
        # return list of log probabilities
        P_log = []

        n_prob = 0
        y_n = ''
        y_n_1 = ''

        # get yn y(n-1)
        for u in K if n > 1 else {'*'}:
            for v in K:
                if pi[(n, u, v)] * q[(u, v, "STOP")] > n_prob:
                    n_prob = pi[(n, u, v)] * q[(u, v, "STOP")]
                    y_n = u
                    y_n_1 = v
        v = y_n_1
        u = y_n
        p_n_1 = math.log(n_prob, 2)
        p_n = math.log(pi[(n, u, v)], 2)
        Y.append(v)
        Y.append(u)
        P_log.append(p_n_1)
        P_log.append(p_n)

        # get all ys in Y through the backpointer
        for k in range(n, 1, -1):
            w = bp[(k, u, v)]
            Y.append(w)
            log_p = math.log(pi[(k - 1, w, u)], 2)
            P_log.append(log_p)
            v = u
            u = w

        Y.reverse()
        P_log.reverse()

        # output the tags for the sentence
        for count, word in enumerate(s_org):
            output.write(str(word) + ' ' + str(Y[count + 1]) + ' ' + str(P_log[count]) + '\n')
        output.write('\n')

if __name__ == "__main__":
    with open("../output/ner_with_rare.counts", "r") as counts:
        with open("../input/ner_dev.dat", "r") as ner_dev:
            with open("../output/sen_tagged.predict", "w+") as output:
                viterbi_tagger(counts, ner_dev, output)
