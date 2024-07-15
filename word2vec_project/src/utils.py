import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

np.random.seed(1)

class DataReader:
    
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFile, min_cnt):

        self.negatives = dict()
        self.discards = dict()
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.word_freq = dict()
        self.token_cnt = 0

        self.inputFile = inputFile
        self.preprocess(min_cnt)
        self.negativeSampling()
        self.subSampling()
    
    def preprocess(self, min_cnt):
        
        word_frequency = dict()
        with open(self.inputFile) as f:
            line = f.readline
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_cnt += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1




        wid = 0
        for w, c in word_frequency.items():
            if c < min_cnt:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_freq[wid] = c
            wid += 1
        

    def subSampling(self):
        t = 1e-5
        self.discards = {word: 1 - np.sqrt(t / freq / self.token_cnt) for word, freq in self.word_freq.items}


    def negativeSampling(self):
        freq_list = np.array(list(np.array(self.word_freq).values())) ** 0.75
        ratio_list = freq_list / np.sum(freq_list)
        cnt_list = ratio_list *  DataReader.NEGATIVE_TABLE_SIZE
        
            

class skipGramDataset(Dataset):
    def __init__(self, data, context_size):
        return
    
    def __len__(self):
        return
    
    def __getitem__(self, index):
        return super().__getitem__(index)




    
