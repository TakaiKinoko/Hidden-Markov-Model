#! /usr/bin/python

__author__="Fang Han Cabrera <fh643@nyu.edu>"
__date__ ="$Dec 1, 2019"

import math
import sys

def compute_log_q(count_file):
    bigram_count_dict, trigram_count_dict = trigram_and_bigram_count(count_file)
    trigram_log_q = dict()
    for trigram in trigram_count_dict.keys():
        bigram = (trigram[0], trigram[1])
        log_q = math.log(trigram_count_dict[trigram] / bigram_count_dict[bigram], math.e)
        trigram_log_q[trigram] = round(log_q, 4)
    return trigram_log_q


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

    # count_file.close()
    return bigram_count_dict, trigram_count_dict


count_file = open("../data/ner_with_rare.counts")

#bigram_count_dict, trigram_count_dict = trigram_and_bigram_count(count_file)
trigram_log_q = compute_log_q(count_file)
count_file.close()

def compute_trigram_q(count_file, trigram_file, trigram_q_file):
    bigram_count_dict, trigram_count_dict = trigram_and_bigram_count(count_file)
    l = trigram_file.readline()
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
            log_q = -sys.maxsize
        else:
            log_q = math.log(q, math.e)
        trigram_q_file.write(l.strip() + " " + str(round(log_q, 4)) + "\n")
        l = trigram_file.readline()

    trigram_q_file.close()
    trigram_file.close()
    
count_file = open("../output/ner_with_rare.counts", "r")
trigram_file = open("../input/trigrams_test.dat", "r")
trigram_q_file = open("../output/trigram.predict","w+")

compute_trigram_q(count_file, trigram_file, trigram_q_file)
