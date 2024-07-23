import os
import re
from collections import Counter
from typing import List, Tuple, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class DataReader:
    def __init__(self, inputfile: str, reverse: bool = False):
        self.inputfile = inputfile
        self.reverse = reverse
        self.data = self._read_data()
        self.word2index, self.index2word, self.vocab_size = self._build_vocab()
        self.SOS_token = self.word2index['<sos>']
        self.EOS_token = self.word2index['<eos>']

    def _read_data(self) -> List[Tuple[str, str]]:
        """
        파일에서 데이터를 읽어와 전처리합니다.
        """
        with open(self.inputfile, 'r', encoding='utf-8') as f:
            data = f.readlines()
        
        data = [self._preprocess_pair(line) for line in data]
        return data

    def _preprocess_pair(self, line: str) -> Tuple[str, str]:
        """
        영어-프랑스어 문장 쌍을 전처리합니다.
        """
        eng, fra = line.split('\t')
        if self.reverse:
            return self._preprocess(fra), self._preprocess(eng)
        else:
            return self._preprocess(eng), self._preprocess(fra)

    def _preprocess(self, sentence: str) -> str:
        """
        문장을 전처리합니다.
        """
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
        sentence = sentence.strip()
        return sentence

    def _build_vocab(self) -> Tuple[Dict[str, int], Dict[int, str], int]:
        """
        단어 사전을 구축합니다.
        """
        counter = Counter()
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for eng, fra in self.data:
            counter.update(eng.split())
            counter.update(fra.split())
        
        # 단어 집합을 정렬하여 인덱스를 부여합니다.
        vocab = special_tokens + sorted(counter.keys())
        word2index = {word: idx for idx, word in enumerate(vocab)}
        index2word = {idx: word for idx, word in enumerate(vocab)}
        vocab_size = len(vocab)
        
        return word2index, index2word, vocab_size

    def get_data(self) -> List[Tuple[str, str]]:
        """
        전처리된 데이터를 반환합니다.
        """
        return self.data

    def get_vocab_size(self) -> int:
        """
        어휘 사전의 크기를 반환합니다.
        """
        return self.vocab_size

    def get_word2index(self) -> Dict[str, int]:
        """
        단어에서 인덱스로의 매핑을 반환합니다.
        """
        return self.word2index

    def get_index2word(self) -> Dict[int, str]:
        """
        인덱스에서 단어로의 매핑을 반환합니다.
        """
        return self.index2word
    
class Seq2SeqDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], word2index: Dict[str, int], max_len: int = 50):
        self.data = data
        self.word2index = word2index
        self.max_len = max_len
        self.SOS_token = word2index['<sos>']
        self.EOS_token = word2index['<eos>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eng_sentence, fra_sentence = self.data[idx]
        eng_indices = [self.SOS_token] + [self.word2index.get(word, self.word2index["<unk>"]) for word in eng_sentence.split()] + [self.EOS_token]
        fra_indices = [self.SOS_token] + [self.word2index.get(word, self.word2index["<unk>"]) for word in fra_sentence.split()] + [self.EOS_token]
        
        eng_indices = eng_indices[:self.max_len]  # 길이 제한
        fra_indices = fra_indices[:self.max_len]  # 길이 제한

        return torch.tensor(eng_indices, dtype=torch.long), torch.tensor(fra_indices, dtype=torch.long)

def collate_fn(batch):
    eng_batch, fra_batch = zip(*batch)
    eng_batch = pad_sequence(eng_batch, batch_first=True, padding_value=0)
    fra_batch = pad_sequence(fra_batch, batch_first=True, padding_value=0)
    return eng_batch, fra_batch

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, '..', 'dataset', 'eng-fra.txt')    
    # DataReader를 사용하여 데이터를 읽어옵니다.
    data_reader = DataReader(input_path, reverse=False)
    data = data_reader.get_data()
    word2index = data_reader.get_word2index()
    index2word = data_reader.get_index2word()

    # Seq2SeqDataset을 생성합니다.
    dataset = Seq2SeqDataset(data, word2index)

    # DataLoader를 생성합니다.
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 예시: DataLoader에서 배치 하나를 가져와 봅니다.
    for eng_batch, fra_batch in dataloader:
        print("English batch:", eng_batch)
        print("French batch:", fra_batch)
        break
