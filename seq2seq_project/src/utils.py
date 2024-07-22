import numpy as np
import numpy as np
from torch.utils.data import Dataset

class DataReader:
    def __init__(self, inputfile):
        self.inputfile = inputfile
        self.word2idx = dict()
        self.word2cnt = dict()
        self.idx2word = {0: "SOS", 1: "EOS"}
        self.n_words = 0
        
        self.read_data()
        self.preprocess()
        

    def read_data(self):
        for line in open(self.inputfile):
            line = line.split()
            if len(line) > 1:
                self.preprocess(self, line)


    def preprocess(self, line):
        
        


                    
                
        


class Seq2SeqDataset(Dataset):
    
