from typing import Tuple, List, Union, Optional
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T

from ezspeech.modules.dataset.utils.text import tokenize
from ezspeech.utils.common import load_dataset, time_reduction



def collate_asr_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    pad_id=0
    wavs = [b[0][0] for b in batch]
    wav_lengths = [torch.tensor(len(f)) for f in wavs]

    tokens = [b[1] for b in batch]
    tokens_lengths = [torch.tensor(len(t)) for t in tokens]

    max_audio_len = max(wav_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    new_audio_signal, new_tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in zip(wavs,wav_lengths,tokens,tokens_lengths):
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



class SpeechRecognitionDataset(Dataset):
    def __init__(
        self,
        filepaths,
        vocab_file,
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
        self.vocab = open(vocab_file,encoding="utf-8").read().splitlines()

        self.dataset = load_dataset(filepaths)

        self.audio_augment= []
        self.augmentation_cfg = augmentation
        if augmentation:
            self.audio_augment= [instantiate(cfg) for cfg in self.augmentation_cfg.values()]
        self.resampler = {
            8000: T.Resample(8000, 16000),
            24000: T.Resample(24000, 16000),
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        transcript = data["transcript"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)
        if sample_rate != 16000:
            if sample_rate not in self.resampler.keys():
                self.resampler[sample_rate] = T.Resample(sample_rate, 16000)
            speech = self.resampler[sample_rate](speech)
            sample_rate = 16000


        tokens=data.get("tokenized_transcript",tokenize(transcript, self.vocab))

        tokens = [self.vocab.index(token) for token in tokens]

        tokens = torch.tensor(tokens, dtype=torch.long)
        return speech, tokens

    def __len__(self) -> int:
        return len(self.dataset)

