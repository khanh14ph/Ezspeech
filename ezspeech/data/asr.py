import torch
from torch.utils.data import Dataset, DataLoader
def tokenize(text, vocab, unk_token='<UNK>', case_sensitive=False):
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    if unk_token not in vocab_dict:
        vocab_dict[unk_token] = len(vocab_dict)
    if not case_sensitive:
        text = text.lower()
    words = text.split()
    tokens = []
    token_indices = [] 
    for word in words:
        if word in vocab_dict:
            token_indices.append(vocab_dict[word])
            tokens.append(word)
        else:
            token_indices.append(vocab_dict[unk_token])
            tokens.append(unk_token)        
    return token_indices, tokens
import pandas as pd
class MelDataset(Dataset):
    def __init__(self, filepath,vocab_file, data_type):
        self.data=pd.read_csv(filepath,sep="\t")
        self.vocab=open(vocab_file).split("\n")[:-1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item=self.data.iloc[idx]
        transcript=item["transcript"]

        return self.data[idx], self.targets[idx]