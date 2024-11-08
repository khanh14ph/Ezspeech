import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T


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


class ASRDataset(Dataset):
    def __init__(self, filepath, vocab_file, data_type=None, feature_type=None):
        self.data = pd.read_csv(filepath, sep="\t")
        self.vocab = open(vocab_file).read().split("\n")[:-1]
        if feature_type != None:
            # https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
            self.feature_extrator = T.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                win_length=None,
                hop_length=512,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                n_mels=256,
                mel_scale="htk",
            )

    def __len__(self):
        return len(self.data)

    def extract_audio_feature(self, x):
        res = self.feature_extrator(x)
        res = res.squeeze().t()
        return res

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        speech, sr = torchaudio.load(item["audio_filepath"])
        audio_feature = self.extract_audio_feature(speech)

        transcript = item["transcript"]
        tokenized_transcript = tokenize(transcript, self.vocab)
        transcript_ids = torch.tensor([self.vocab.index(token) for token in tokenized_transcript],dtype=torch.long)
        return audio_feature, transcript_ids


def collate_asr(batch):
    audio_features = [i[0] for i in batch]
    audio_feature_length = [len(i) for i in audio_features]
    audio_features = pad_sequence(audio_features, batch_first=True)
    print("audio_features",audio_features.shape)
    transcript_ids = [i[1] for i in batch]
    transcript_ids_length = [len(i) for i in audio_features]
    print("transcript_ids",transcript_ids)
    transcript_ids = pad_sequence(transcript_ids, batch_first=True)
    
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
        print(m.shape)
