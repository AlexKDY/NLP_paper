import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.center_embeddings = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embed_size, sparse=True)

        init_range = 1.0 / embed_size
        init.unform(self.u_embeddings.weight.data, -init_range, init_range)
        init.constant(self.v_embeddings.weight.data, 0)

    def forward(self, center, pos_c, neg_c):
        embed_center = self.center_embeddings(embed_center)
        embed_pos = self.context_embeddings(pos_c)
        embed_neg = self.context_embeddings(neg_c)

        score = torch.sum(torch.mul(embed_center, embed_pos), dim=1)
        score = torch.clamp(score, max = 10, min = -10)
        score = -F.logsigmoid(score)


        neg_score = torch.bmm(embed_neg, embed_center.unsqueeze(2)).squeez()
        neg_score = torch.clamp(neg_score, max = 10, min = -10)
        neg_score = -torch.sum(F.sigmoid(-neg_score), dim = 1)

        return torch.mean(score + neg_score)
    
    def save_embedding(self, id2word, file_name):
        embedding = self.center_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))