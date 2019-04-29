# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:29:25 2018

@author: moamin
"""
import sys, os
import numpy as np
import gzip
import tqdm
import bisect

seq_convert = {'N':'0', 'A':'1', 'C':'2', 'G':'3', 'T':'4'}
#print seq_convert

file_list = []
for x in os.listdir('.'):
    if not x.endswith("gz"):
        continue
    file_list.append(x)
    #print x

#inframe_write = open("coding_" + sys.argv[1] + "_codon.txt", "w")
non_orf_write = open("non_genic_orf_neg_" + sys.argv[1] + "_codon.txt", "w")
orf_write = open("genic_orf_neg_" + sys.argv[1] + "_codon.txt", "w")
pos_write = open("pos_" + sys.argv[1] + "_codon.txt", "w")

file_list = ['dataset/Predict_Bothways/forward_genes/forward_batch_0_0.gz']
codon_type = sys.argv[1]
window_size = int(sys.argv[2])

for file_name in file_list:
    print (file_name)

    end_set = set()
    start_set = set()

    # start_pos = set()
    # start_neg = set()

    orf_start_neg = set()
    non_orf_start_neg = set()

    end_pos = set()

    orf_end_neg = set()
    non_orf_end_neg = set()

    start_pos_set = set()
    end_pos_set = set()

    start_pos = 0
    start_neg = 0

    end_pos = 0
    end_neg = 0

    file_read = gzip.open(file_name, "rt", encoding='utf-8')

    
    seq_name = file_read.readline()
    print (seq_name)
    
    sequence = 'N'*50 + file_read.readline().strip() + 'N'*50
    gene_nogene = '0'*50 + file_read.readline().strip() + '0'*50
    start_codon = '0'*50 + file_read.readline().strip() + '0'*50
    stop_codon = '0'*50 + file_read.readline().strip() + '0'*50

    assert(len(sequence) == len(gene_nogene))
    assert(len(sequence) == len(start_codon))
    assert(len(start_codon) == len(stop_codon))

    """
    for line in file_read:
        print "", len(line)
    #break
    """

    start_codon_set = set(["ATG", "GTG", "TTG", "CTG", "ATT", "ATC"])
    end_codon_set = set(["TAG", "TAA", "TGA"])

    print("Calculate ORFs")
    original_start_index = []
    original_stop_index = []
    for i in tqdm.tqdm(range(len(sequence)-2)):
        codon = sequence[i:i+3]
        if codon in start_codon_set:
            original_start_index += [i]
        elif(codon in end_codon_set):
            original_stop_index += [i]

    fragment_interval = []
    length_dist = []
	
    for x in tqdm.tqdm(range(0, len(original_start_index))):
        stop_index = bisect.bisect_left(original_stop_index, original_start_index[x])

        for y in range(stop_index, len(original_stop_index)):
            i = original_stop_index[y]
            k = original_start_index[x]            

            if i - k > 25000:
                break

            if not (i-k) % 3 == 0:
                continue

            fragment_interval += [[k, i]]
            length_dist += [i-k]
            break
        
    print("Calculate intervals")
    for interval in tqdm.tqdm(fragment_interval): 
        start, end = interval[0], interval[1]

        around_start = sequence[start-49 : start+52]
        around_end = sequence[end-49 : end+52]

        if((codon_type == 'start') and not(start in start_set)):
            if start_codon[start] == '1':
                start_pos_set.add(start)
                start_pos += 1
            else:
                around_gene_boundary = [int(x) for x in start_codon[start-window_size : start+window_size]]
                if np.sum(around_gene_boundary) == 0:
                    non_orf_start_neg.add(start)
                else:
                    orf_start_neg.add(start)
                start_neg += 1

            start_set.add(start)
        elif((codon_type == 'end') and not(end in end_set)):
            if stop_codon[end] == '1':
                end_pos_set.add(end)
                end_pos += 1
            else:
                around_gene_boundary = [int(x) for x in stop_codon[end-window_size : end+window_size]]
                if np.sum(around_gene_boundary) == 0:
                    non_orf_end_neg.add(end)
                else:
                    orf_end_neg.add(end)
                end_neg += 1
            end_set.add(end)

    print(start_pos, start_neg, len(orf_start_neg), len(non_orf_start_neg), end_pos, end_neg)

    # Sampling the non orf
    num_neg = len(orf_start_neg)
    import random

    if(codon_type == 'start'):
        non_orf_start_neg_samples = random.sample(list(non_orf_start_neg), num_neg)
        for p in start_pos_set:
            train_seq = sequence[p-49 : p+52]
            convert = ''.join([seq_convert[train_seq[k]] for k in range(len(train_seq))])
            pos_write.write("{}\n".format(convert))
        for p in orf_start_neg:
            train_seq = sequence[p-49 : p+52]
            convert = ''.join([seq_convert[train_seq[k]] for k in range(len(train_seq))])
            orf_write.write("{}\n".format(convert))
        for p in non_orf_start_neg_samples:
            train_seq = sequence[p-49 : p+52]
            convert = ''.join([seq_convert[train_seq[k]] for k in range(len(train_seq))])
            non_orf_write.write("{}\n".format(convert))
    else:
        non_orf_end_neg_samples = random.sample(list(non_orf_end_neg), len(orf_end_neg))
        for p in end_pos_set:
            train_seq = sequence[p-49 : p+52]
            convert = ''.join([seq_convert[train_seq[k]] for k in range(len(train_seq))])
            pos_write.write("{}\n".format(convert))
        for p in orf_end_neg:
            train_seq = sequence[p-49 : p+52]
            convert = ''.join([seq_convert[train_seq[k]] for k in range(len(train_seq))])
            orf_write.write("{}\n".format(convert))
        for p in non_orf_end_neg_samples:
            train_seq = sequence[p-49 : p+52]
            convert = ''.join([seq_convert[train_seq[k]] for k in range(len(train_seq))])
            non_orf_write.write("{}\n".format(convert))



    file_read.close()

print ("done")
pos_write.close()
orf_write.close()
non_orf_write.close()
