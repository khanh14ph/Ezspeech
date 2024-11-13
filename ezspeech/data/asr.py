import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
from ezspeech.data.augment import time_masking, frequency_masking

def tokenize(text, vocab, blank_idx=0, word_delimiter_idx=1, unk_idx=2):
    text = text.replace(" ", "|")
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    tokens = []
    while text:
        matched = False

        for subword in sorted_vocab:
            if text.startswith(subword):
                tokens.append(subword)
                text = text[len(subword) :]
                matched = True
                break
        if not matched:
            tokens.append(vocab[unk_idx])
            text = text[1:]

    return tokens

import random
class ASRDataset(Dataset):
    def __init__(self, filepath, vocab_file, data_type=None, feature_type=None):
        self.data = pd.read_csv(filepath, sep="\t")
        self.vocab = open(vocab_file).read().split("\n")[:-1]
        if feature_type != None:
            sr=16000
            # https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
            self.feature_extrator = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=int(0.05*sr),
                win_length=int(0.025*sr),
                hop_length=int(0.01*sr),
                center=False,
                n_mels=128,
            )
        self.augment=[time_masking(0.3),frequency_masking(0.3)]
    def __len__(self):
        return len(self.data)

    def extract_audio_feature(self, x):
        res = self.feature_extrator(x)
        res = res.squeeze(0).clamp(1e-5).log()
        return res

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        speech, sr = torchaudio.load(item["audio_filepath"])
        audio_feature = self.extract_audio_feature(speech)
        for i in self.augment:
            audio_feature=i(audio_feature)


        transcript = item["transcript"]
        tokenized_transcript = tokenize(transcript, self.vocab)
        transcript_ids = torch.tensor(
            [self.vocab.index(token) for token in tokenized_transcript],
            dtype=torch.long,
        )
        return audio_feature.t(), transcript_ids


def collate_asr(batch):
    audio_features = [i[0] for i in batch]
    audio_feature_length = torch.tensor([len(i) for i in audio_features],dtype=torch.long)
    audio_features = pad_sequence(audio_features, batch_first=True)
    transcript_ids = [i[1] for i in batch]
    transcript_ids_length = torch.tensor([len(i) for i in transcript_ids],dtype=torch.long)
    transcript_ids = pad_sequence(transcript_ids, batch_first=True)
    # print("audio_feature_length",audio_feature_length)
    # print("transcript_ids_length",transcript_ids_length)
    return audio_features, audio_feature_length, transcript_ids, transcript_ids_length


if __name__ == "__main__":
    dataset = ASRDataset(
        "/home/msi/Documents/Ezspeech/train.tsv",
        "/home/msi/Documents/Ezspeech/ezspeech/resources/vocab.txt",
        feature_type="Mel",
    )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)
    for b in data_loader:
        m, n, p, q = b
