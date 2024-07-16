import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DataReader, skipGramDataset
from model import SkipGram

class ModelTrainer:
    def __init__(self, input_file, output_file, embed_size=100, batch_size=32, context_size=5, iter=3,
                 lr = 0.001, min_cnt = 12):
        
        self.data = DataReader(input_file, min_cnt)
        dataset = skipGramDataset(self.data, context_size)
        self.dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=2, collate_fn=dataset.collate)

        self.output_file = output_file
        self.vocab_size = len(self.data.word2id)
        self.embed_size = embed_size
        self.iter = iter
        self.lr = lr
        self.model = SkipGram(self.vocab_size, self.embed_size)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()

    def train(self):

        for iter in range(self.iter):

            print("\n\n\nIteration: " + str(iter + 1))
            optimizer = optim.SparseAdam(self.model.parameters(), lr = self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for idx, batch_sample in enumerate(tqdm(self.dataloader)):

                if (len(batch_sample[0]) > 1):
                    center = batch_sample[0].to(self.device)
                    pos = batch_sample[1].to(self.device)
                    neg = batch_sample[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.model.forward(center, pos, neg)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if idx > 0 and idx % 500 == 0:
                        print(" Loss: " + str(running_loss))

                self.model.save_embedding(self.data.id2word, self.output_file)


import os

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, '..', 'dataset', 'text8')
    output_path = os.path.join(current_dir, '..', 'models', 'output.vec')
    skipgram = ModelTrainer(input_path, output_path)
    skipgram.train()



