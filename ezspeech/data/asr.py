import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import torchaudio.functional as F
import torchaudio.transforms as T

def tokenize(text, vocab,blank_idx=0, word_delimiter_idx=1,unk_idx=2):
    text=text.replace(" ","|")
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    tokens = []
    while text:
        matched = False

        for subword in sorted_vocab:
            if text.startswith(subword):
                tokens.append(subword)
                text = text[len(subword):]
                matched = True
                break
        if not matched:
            tokens.append(vocab[unk_idx])
            text = text[1:]
    
    return tokens

class MelDataset(Dataset):
    def __init__(self, filepath,vocab_file, data_type,feature_type=None):
        self.data=pd.read_csv(filepath,sep="\t")
        self.vocab=open(vocab_file).split("\n")[:-1]
        if feature_type!=None:  
            self.feature_extrator=T.Spectrogram(n_fft=512, hop_length=512/4)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item=self.data.iloc[idx]
        speech,sr=torchaudio.load(item["audio_filepath"])
        audio_feature=self.feature_extrator(speech)
        
        transcript=item["transcript"]
        tokenized_transcript=tokenize(transcript)
        transcript_ids=[self.vocab.index(token) for token in tokenized_transcript]
        return self.data[idx], transcript_ids
if __name__=="__main__":
    vocab=open("/home/msi/Documents/Ezspeech/ezspeech/resources/vocab.txt").read().split("\n")[:-1]
    a=tokenize("a b ccccdeprrt",vocab)
    print(a)
