#! /usr/bin/python

__author__="Fang Han Cabrera <fh643@nyu.edu>"
__date__ ="$Nov 22, 2019"

import count_freqs
import sys
from collections import defaultdict

def find_rares (cnts_file):
    '''
    find all <word, tag> pairs that have a count less than 5 and return them in a list
    '''
    rares = set() # list to store all rare words found in ner.counts
    cnt_dict = defaultdict(int)

    with open(cnts_file) as fp: 
        for line in fp: 
            fields = line.strip().split(" ")
            tag = fields[1]
            cnt = (int)(fields[0])
            if tag == 'WORDTAG': # pick out unigrams with a count less than 5
                cnt_dict[fields[-1]] += int(cnt)
        fp.close()
    for k in cnt_dict.keys():
        if cnt_dict[k] < 5:
            rares.add(k)
    return rares

def retag_with_rare (rares, corpus, outpath):
    '''
    replace the original tags in the training data with RARE and write to training data file 
    '''
    fout = open(outpath, 'w+')
    corpus_file = open(corpus, "r")
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            fields = line.split(" ")
            if fields[0] == '':
                fout.write(l)
            elif fields[0] in rares:
                fout.write('_RARE_ ' + fields[1] + '\n')
            else:
                fout.write(l)
            '''
            word = " ".join(fields[:-1]) 
            tag = fields[-1]
            if (word, tag) in rares: 
                fout.write("_RARE_ " + tag + '\n')
            else: 
                fout.write(l + '\n')'''
        else:
            fout.write(l)
        l = corpus_file.readline()
    return


if __name__ == "__main__":
    orig_counts = "../output/ner.counts"
    train_data = "../input/ner_train.dat"
    new_counts = '../output/ner_with_rare.counts'
    new_train_data = '../output/ner_train_with_rare.dat'

    rares = find_rares(orig_counts)
    retag_with_rare(rares, train_data, new_train_data)
