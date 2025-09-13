from typing import List, Optional, Tuple, Union

import torch
import torchaudio
import torchaudio.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ezspeech.modules.data.utils.text import Tokenizer
from ezspeech.utils.common import load_dataset, time_reduction


class SpeechRecognitionDataset(Dataset):
    def __init__(
        self,
        filepaths,
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechRecognitionDataset, self).__init__()

        self.dataset = load_dataset(filepaths)

        self.audio_augment = []
        self.augmentation_cfg = augmentation
        if augmentation:
            self.audio_augment = [
                instantiate(cfg) for cfg in self.augmentation_cfg.values()
            ]
        self.resampler = {
            8000: T.Resample(8000, 16000),
            24000: T.Resample(24000, 16000),
        }

    def set_tokenizer(self, tokenzier):
        self.tokenizer = tokenzier

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        transcript = data["text"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)
        if sample_rate != 16000:
            if sample_rate not in self.resampler.keys():
                self.resampler[sample_rate] = T.Resample(sample_rate, 16000)
            speech = self.resampler[sample_rate](speech)
            sample_rate = 16000

        tokens = self.tokenizer.encode(transcript)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return speech, tokens

    def __len__(self) -> int:
        return len(self.dataset)
    def get_dur(self, idx: int) -> int:
        return self.dataset[idx]["duration"]
    def collate_asr_data(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        pad_id = 0
        wavs = [b[0][0] for b in batch]
        wav_lengths = [torch.tensor(len(f)) for f in wavs]

        tokens = [b[1] for b in batch]
        tokens_lengths = [torch.tensor(len(t)) for t in tokens]

        max_audio_len = max(wav_lengths).item()
        max_tokens_len = max(tokens_lengths).item()

        new_audio_signal, new_tokens = [], []
        for sig, sig_len, tokens_i, tokens_i_len in zip(
            wavs, wav_lengths, tokens, tokens_lengths
        ):
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            new_audio_signal.append(sig)
            if tokens_i_len < max_tokens_len:
                pad = (0, max_tokens_len - tokens_i_len)
                tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
            new_tokens.append(tokens_i)

        new_audio_signal = torch.stack(new_audio_signal)
        audio_lengths = torch.stack(wav_lengths)

        new_tokens = torch.stack(new_tokens)
        tokens_lengths = torch.stack(tokens_lengths)

        return new_audio_signal, audio_lengths, new_tokens, tokens_lengths