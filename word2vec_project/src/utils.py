import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

np.random.seed(1)

class DataReader:
    
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFile, min_cnt):

        self.negatives = []
        self.discards = dict()
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.word_freq = dict()
        self.token_cnt = 0
        self.sentece_cnt = 0

        self.inputFile = inputFile
        self.preprocess(min_cnt)
        self.negativeSampling()
        self.subSampling()
    
    def preprocess(self, min_cnt):
        
        word_frequency = dict()
        with open(self.inputFile) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                if len(line) > 1:
                    self.sentence_cnt += 1
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
        f = np.array(list(self.word_frequency.values())) / self.token_cnt
        self.discards = 1 - np.sqrt(t/f)


    def negativeSampling(self):
        freq_list = np.array(list(np.array(self.word_freq).values())) ** 0.75
        ratio_list = freq_list / np.sum(freq_list)
        cnt_list = np.round(ratio_list *  DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(cnt_list):
            self.negatives =[wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegativeSample(self, target, size):
        neg_sample = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        neg_sample = [neg for neg in neg_sample if neg not in target]
        while len(neg_sample) < size:
            additional = self.negatives[self.negpos: self.negpos + (size - len(neg_sample))]
            self.negpos = (self.negpos + len(additional)) % len(self.negatives)
            additional = [neg for neg in additional if neg not in target]
            neg_sample.extend(additional)

        return neg_sample

        
class skipGramDataset(Dataset):
    def __init__(self, data, context_size):
        self.data = data
        self.context_size = context_size
        self.input_file = open(data.inputFile)
    
    def __len__(self):
        return self.data.sentence_cnt
    
    def __getitem__(self, index):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    sampled_ids = [self.data.word2id[word] for word in words if word in self.data.word2id.keys() and np.random.rand() > self.data.discards[self.word2id[word]]]
                
                    range = np.random.randint(1, self.context_size)                        
                    
                    return [(center, context, self.data.getNegativeSample([center, context], 5)) for c_id, center in enumerate(sampled_ids)
                            for context in sampled_ids[max(0, c_id - range): min(c_id + range, len(sampled_ids))] if center != context]
                
    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)





    
