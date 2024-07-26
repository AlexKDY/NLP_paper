import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DataReader, Seq2SeqDataset
from model import EncoderRNN, DecoderRNN, Seq2Seq

class ModelTrainer:
    def __init__(self, input_path, hidden_size = 1000, batch_size = 128, iter = 5, reverse = False):

        self.data = DataReader(input_path, reverse)
        dataset = Seq2SeqDataset(self.data.get_data(), self.data.get_word2index())
        self.dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()
        
        vocab_size = self.data.get_vocab_size()
        self.encoder = EncoderRNN(vocab_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, )
        self.model = Seq2Seq(self.encoder, self.decoder, self.device)



if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, '..', 'dataset', 'eng-fra.txt')    
