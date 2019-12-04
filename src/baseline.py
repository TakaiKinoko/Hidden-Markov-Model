#! /usr/bin/python

__author__ = "Fang Han Cabrera <fh643@nyu.edu>"
__date__ ="$Nov 22, 2019"

import math
import sys

def get_counts (cnts_file):
    '''
    computes count(ner) for all ners
    and the collection of <ner, count> pair for each co-occurring word

    :parameter cnts_file: path of file containing all ner counts
    :return: a pair of dicts
    '''
    # dict to store <ner, count> pairs, aka count(y)
    ner_count = {}
    # dict to store <word, <ner, count> list>, aka <x, list of count(y, x) for all y>
    joint_count = {}

    with open(cnts_file) as fp:
        for line in fp:
            # line format: <cnt> <tag> <ner> <word(s)>
            fields = line.strip().split(" ")
            cnt = (int)(fields[0])
            tag = fields[1]
            ner = fields[2]
            if tag == "WORDTAG":
                word = fields[-1]
                # increment ner count
                if ner in ner_count:
                    ner_count[ner] += cnt
                else:
                    ner_count[ner] = cnt
                tmp = joint_count.setdefault(word, {})
                tmp[ner] = cnt
                # update joint_count dict
                joint_count[word] = tmp
    return (ner_count, joint_count)

def compute_emission(word, ner):
    '''
    compute e(x|y) as count(y, x) / count(y)

    :parameter word: word, e.g. 'Nations'
    :parameter ner: name-entity tag, e.g. 'I-ORG'
    :return: log probability of the NER tag for the word
    '''
    cnt_list = joint_count.get(word, joint_count.get("_RARE_"))
    ner_cnt = ner_count.get(ner, 0) # default to 0 if not found
    joint_cnt = cnt_list.get(ner, 0) # default to 0 if not found

    if ner_cnt > 0:
        emission = joint_cnt / ner_cnt
        return math.log(emission, 2)
    else:
        print("Unknown NER encountered")
        sys.exit(1)

def get_max_prob(word):
    '''
    :parameter word: word to make prediction on
    :return: predicted NER and it's log probability
    '''
    cnt_list = joint_count.get(word, joint_count.get("_RARE_"))

    em_max = -sys.maxsize
    prediction = "O"

    for ner in cnt_list.keys():
        emission = compute_emission(word, ner)
        if emission > em_max:
            em_max = emission
            prediction = ner
    return prediction, em_max

def ner_tagger(in_file, prediction_file):
    '''
    A simple name entity tagger that always produces the tag:
             y* = argmax_y e(x|y)
    for each word x.

    :paramter in_file: test data to be tagged
    :prediction_file: file path to write the prediction to
    '''
    fout = open(prediction_file, 'w+')
    data_file = open(in_file, "r")
    l = data_file.readline()
    while l:
        word = l.strip()
        if word: # Nonempty line
            prediction, prob = get_max_prob(word)
            fout.write(word + " " + prediction + " " + str(prob) + '\n')
        else:
            fout.write(l)
        l = data_file.readline()

if __name__ == "__main__":
    new_counts = '../output/ner_with_rare.counts'
    dev_set = '../input/ner_dev.dat'
    prediction_file = '../output/dev_tagged.predict'
    ner_count, joint_count = get_counts(new_counts)
    ner_tagger(dev_set, prediction_file)
