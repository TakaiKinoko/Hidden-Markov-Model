#! /usr/bin/python

__author__="Fang Han Cabrera <fh643@nyu.edu>"
__date__ ="$Dec 1, 2019"

import math
import sys

def compute_trigram_probs(count_file, test_data, prediction):
    '''
    :param count_file: count file computed by replace_with_rare.py
    :param test_data: file containing test_data where each trigram occupies a line
    :param prediction: file to write predicted <trigram>, <log probability> to
    :return: log probability of the trigrams read from trigram_file
    '''
    bigram_count_dict, trigram_count_dict = trigram_and_bigram_count(count_file)
    l = test_data.readline()
    while l:
        trigrams = l.strip().split(" ")
        trigram = (trigrams[0], trigrams[1], trigrams[2])
        bigram = (trigrams[0], trigrams[1])
        trigram_count = trigram_count_dict.get(trigram, 0)
        bigram_count = bigram_count_dict.get(bigram, 0)
        q = 0
        if bigram_count > 0:
            q = trigram_count / bigram_count
        if q == 0:
            log_prob = -sys.maxsize
        else:
            log_prob = math.log(q, math.e)
        prediction.write(l.strip() + " " + str(log_prob) + "\n")
        l = test_data.readline()

def trigram_and_bigram_count(count_file):
    trigram_count_dict = dict()
    bigram_count_dict = dict()
    l = count_file.readline()
    while l:
        fields = l.strip().split(" ")
        if fields[1] == "2-GRAM":
            bigram_count_dict[(fields[-2], fields[-1])] = int(fields[0])
        if fields[1] == "3-GRAM":
            trigram_count_dict[(fields[-3], fields[-2], fields[-1])] = int(fields[0])
        l = count_file.readline()
    return bigram_count_dict, trigram_count_dict

def compute_log_prob(count_file):
    bigram_count_dict, trigram_count_dict = trigram_and_bigram_count(count_file)
    trigram_log_prob = dict()
    for trigram in trigram_count_dict.keys():
        bigram = (trigram[0], trigram[1])
        log_prob = math.log(trigram_count_dict[trigram] / bigram_count_dict[bigram], math.e)
        trigram_log_prob[trigram] = log_prob
    return trigram_log_prob

if __name__ == "__main__":
    with open("../output/ner_with_rare.counts", "r") as count_file:
        with open("../input/trigrams_test.dat", "r") as test_data:
            with open("../output/trigram.predict","w+") as prediction:
                compute_trigram_probs(count_file, test_data, prediction)
                prediction.close()
        test_data.close()
    count_file.close()
