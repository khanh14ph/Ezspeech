import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import random
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
from ezspeech.data.util import extract_audio_feature
from hydra.utils import instantiate
from omegaconf import DictConfig
import re
def tokenize(text, vocab):
    text = text.replace(" ", "|")
    patterns="|".join(map(re.escape,sorted(vocab,reverse=True)))
    tokens=re.findall(patterns,text)
    return tokens

class ASRDataset(Dataset):
    def __init__(self, filepath, vocab_file,augmentations=None):
        self.data = pd.read_csv(filepath, sep="\t")
        self.vocab = open(vocab_file).read().split("\n")[:-1]
        sr=16000
        self.is_train=False
        if augmentations!=None:
            self.is_train=True
            self.feature_augments=[j for i,j  in augmentations.feature.items()]
            self.wav_augment=[j for i,j in augmentations.raw_wav.items()]
        # https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
        self.augment=None
            
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        transcript = item["transcript"]
        tokenized_transcript = tokenize(transcript, self.vocab)
        transcript_ids = torch.tensor(
            [self.vocab.index(token) for token in tokenized_transcript],
            dtype=torch.long,
        )
        speech, sr = torchaudio.load(item["audio_filepath"])
        # if self.is_train:
        #     for i in self.wav_augment:
        #         speech=i(speech)
        # print("speech",speech.shape)

        audio_feature = extract_audio_feature(speech,sr)
        # print("audio_feature",audio_feature.shape)
        # print(item["audio_filepath"])
        # print("audio_feature",audio_feature.shape)
        if self.is_train:
            for i in self.feature_augments:        
                audio_feature=i(audio_feature)
        audio=item["audio_filepath"]
        return audio_feature.t(), transcript_ids,audio


def collate_asr(batch):
    audio_features = [i[0] for i in batch]
    audio_feature_length = torch.tensor([len(i) for i in audio_features],dtype=torch.long)
    audio_features = pad_sequence(audio_features, batch_first=True)
    transcript_ids = [i[1] for i in batch]
    transcript_ids_length = torch.tensor([len(i) for i in transcript_ids],dtype=torch.long)
    transcript_ids = pad_sequence(transcript_ids, batch_first=True)
    # print("audio_features",audio_features.shape)
    audio=[i[2] for i in batch]
    # print("audio1",audio)
    return audio_features, audio_feature_length, transcript_ids, transcript_ids_length,audio


if __name__ == "__main__":
    vocab = open("/home4/khanhnd/Ezspeech/ezspeech/resources/vocab.txt").read().split("\n")[:-1]
    res=tokenize("thụ động",vocab)
    print(res)
    import pandas as pd
    import librosa
    df=pd.read_csv("/home4/khanhnd/vivos/train.tsv",sep="\t")
    for idx,i in df.iterrows():
        transcript=i["transcript"]
        dur=librosa.get_duration(path=i["audio_filepath"])
        transcript_idx=tokenize(transcript,vocab)
        if (dur/0.04)< len(transcript_idx)-3:
            print(i["audio_filepath"])

    # print([vocab.index(token) for token in res])
    a=ASRDataset("/home4/khanhnd/vivos/train.tsv","/home4/khanhnd/Ezspeech/ezspeech/resources/vocab.txt")
    a[0]
