from typing import Tuple, List, Union, Optional
from omegaconf import DictConfig
from tokenizers import Tokenizer

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import torchaudio.transforms as T

from ezspeech.modules.dataset.utils.text import tokenize
from ezspeech.modules.dataset.utils.audio import (
    get_augmentation,
    extract_filterbank,
    extract_melspectrogram,
)
from ezspeech.utils.common import load_dataset, time_reduction


def collate_asr_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    feature_lengths = [len(f) for f in features]
    features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    tokens = [b[1] for b in batch]
    token_lengths = [len(t) for t in tokens]
    tokens = pad_sequence(tokens, batch_first=True)
    token_lengths = torch.tensor(token_lengths, dtype=torch.long)
    # print("features", features.shape)
    # audio_filepaths=[b[2] for b in batch]
    return features, feature_lengths, tokens, token_lengths


class SpeechRecognitionDataset(Dataset):
    def __init__(
        self,
        filepaths,
        vocab,
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechRecognitionDataset, self).__init__()
        sample_rate = 16000
        self.transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(0.05 * sample_rate),
            win_length=int(0.025 * sample_rate),
            hop_length=int(0.01 * sample_rate),
            n_mels=128,
            center=False,
        )
        self.vocab = open(vocab).read().splitlines()

        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        self.augmentation = augmentation
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation
        self.resampler = {
            8000: T.Resample(8000, 16000),
            24000: T.Resample(24000, 16000),
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        transcript = data["transcript"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        if sample_rate != 16000:
            if sample_rate not in self.resampler.keys():
                self.resampler[sample_rate] = T.Resample(sample_rate, 16000)
            speech = self.resampler[sample_rate](speech)
            sample_rate = 16000

        speech = speech.mean(dim=0, keepdim=True)

        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate, self.transformation)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        tokens = tokenize(transcript, self.vocab)

        tokens = [self.vocab.index(token) for token in tokens]

        tokens = torch.tensor(tokens, dtype=torch.long)
        frame_length = feature.shape[1] / 0.01
        if len(tokens) > frame_length:
            qweqwe
        return feature.t(), tokens, audio_filepath

    def __len__(self) -> int:
        return len(self.dataset)


class SpeechClassificationDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        labels: List[str],
        augmentation: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.labels = labels
        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        label = data["label"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        label = self.labels.index(label)
        label = torch.tensor(label, dtype=torch.long)

        return feature.t(), label

    def __len__(self) -> int:
        return len(self.dataset)


def collate_wav_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    feature_lengths = [len(feat) for feat in features]
    features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    audio_tgts = [b[1] for b in batch]
    audio_lens = [len(audio) for audio in audio_tgts]
    audio_tgts = pad_sequence(audio_tgts, batch_first=True).transpose(1, 2)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)

    return features, feature_lengths, audio_tgts, audio_lens
