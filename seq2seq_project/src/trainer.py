import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DataReader, Seq2SeqDataset
from model import EncoderRNN, DecoderRNN, Seq2Seq

class ModelTrainer:
    def __init__(self, input_path, hidden_size = 1000, batch_size = 128, iter = 7.5, reverse = False):

        self.data = DataReader(input_path, reverse)
        dataset = Seq2SeqDataset(self.data.get_data(), self.data.get_word2index())
        self.dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        vocab_size = self.data.get_vocab_size()
        self.encoder = EncoderRNN(vocab_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, )
        self.model = Seq2Seq(self.encoder, self.decoder, self.device)
        self.iter = iter

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.7)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.data.get_word2index()['<pad>'])

        
        if self.use_cuda:
            self.model.cuda()
    
    def adjust_lr(self, optimizer, epoch, iter, total_iter):
        if epoch >=5:
            lr = 0.7 * (0.5 ** ((epoch - 5) * 2 + iter / total_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def train(self):
        self.model.train()
        total_iter = len(self.dataloader)

        for epcoh in range(int(self.iter)):
            epoch_loss = 0
            for i, (src, trg, src_len, trg_len) in enumerate(tqdm(self.dataloader)):
                src, trg = src.to(self.device), trg.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(src, src_len, trg, trg_len)
                output_dim = output.shape[-1]

                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = self.criterion()


if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, '..', 'dataset', 'eng-fra.txt')    
