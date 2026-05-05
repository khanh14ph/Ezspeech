from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import torch
import torch.nn as nn
import librosa
import torchaudio
import torchaudio.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ezspeech.modules.data.utils.text import Tokenizer
from ezspeech.utils.common import load_jsonl, time_reduction


class SpeechRecognitionDataset(Dataset):
    def __init__(
        self,
        filepaths,
        augmentation: Optional[DictConfig] = None,
        
        data_dir="",
        max_duration=float("inf"),
        min_duration=0.0,
    ):
        super(SpeechRecognitionDataset, self).__init__()

        self.dataset = load_jsonl(filepaths,data_dir=data_dir)
        print("max_duration: ", max_duration)
        self.dataset = [d for d in tqdm(self.dataset) if min_duration <= d["duration"] <= max_duration]
        self.data_dir=data_dir
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

        speech, sample_rate = librosa.load(audio_filepath, sr=None)
        speech=torch.from_numpy(speech).unsqueeze(0)
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

class SpeechRecognitionDatasetSC(Dataset):
    def __init__(
        self,
        filepaths,
        augmentation: Optional[DictConfig] = None,
        data_dir="",
        max_duration=float("inf"),
        min_duration=0.0,
    ):
        super(SpeechRecognitionDatasetSC, self).__init__()

        self.dataset = load_jsonl(filepaths)
        self.dataset = [d for d in tqdm(self.dataset) if min_duration <= d["duration"] <= max_duration]
        self.data_dir=data_dir
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

    def set_tokenizer(self, tokenizer_grapheme, tokenizer_phoneme):
        self.tokenizer_grapheme = tokenizer_grapheme
        self.tokenizer_phoneme = tokenizer_phoneme

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = self.data_dir+data["audio_filepath"]
        transcript = data["text"]
        transcript_ipa=data["text_ipa"]
        speech, sample_rate = torchaudio.load(audio_filepath)
        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)
        if sample_rate != 16000:
            if sample_rate not in self.resampler.keys():
                self.resampler[sample_rate] = T.Resample(sample_rate, 16000)
            speech = self.resampler[sample_rate](speech)
            sample_rate = 16000

        tokens_grapheme = self.tokenizer_grapheme.encode(transcript)
        tokens_grapheme = torch.tensor(tokens_grapheme, dtype=torch.long)

        tokens_phoneme = self.tokenizer_phoneme.encode(transcript_ipa)
        tokens_phoneme = torch.tensor(tokens_phoneme, dtype=torch.long)

        return speech, tokens_grapheme, tokens_phoneme

    def __len__(self) -> int:
        return len(self.dataset)
    def get_dur(self, idx: int) -> int:
        return self.dataset[idx]["duration"]
    def collate_asr_data(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        pad_id = 0
        wavs = [b[0][0] for b in batch]
        wav_lengths = [torch.tensor(len(f)) for f in wavs]

        tokens_grapheme = [b[1] for b in batch]
        tokens_grapheme_lengths = [torch.tensor(len(t)) for t in tokens_grapheme]

        tokens_phoneme = [b[2] for b in batch]
        tokens_phoneme_lengths = [torch.tensor(len(t)) for t in tokens_phoneme]

        max_audio_len = max(wav_lengths).item()
        max_tokens_grapheme_len = max(tokens_grapheme_lengths).item()
        max_tokens_phoneme_len = max(tokens_phoneme_lengths).item()

        new_audio_signal, new_tokens_grapheme, new_tokens_phoneme = [], [], []
        for sig, sig_len, tokens_grapheme_i, tokens_grapheme_i_len, tokens_phoneme_i, tokens_phoneme_i_len in zip(
            wavs, wav_lengths, tokens_grapheme, tokens_grapheme_lengths, tokens_phoneme, tokens_phoneme_lengths
        ):
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            new_audio_signal.append(sig)
            if tokens_grapheme_i_len < max_tokens_grapheme_len:
                pad = (0, max_tokens_grapheme_len - tokens_grapheme_i_len)
                tokens_grapheme_i = torch.nn.functional.pad(tokens_grapheme_i, pad, value=pad_id)
            new_tokens_grapheme.append(tokens_grapheme_i)
            if tokens_phoneme_i_len < max_tokens_phoneme_len:
                pad = (0, max_tokens_phoneme_len - tokens_phoneme_i_len)
                tokens_phoneme_i = torch.nn.functional.pad(tokens_phoneme_i, pad, value=pad_id)
            new_tokens_phoneme.append(tokens_phoneme_i)

        new_audio_signal = torch.stack(new_audio_signal)
        audio_lengths = torch.stack(wav_lengths)

        new_tokens_grapheme = torch.stack(new_tokens_grapheme)
        new_tokens_phoneme = torch.stack(new_tokens_phoneme)

        tokens_grapheme_lengths = torch.stack(tokens_grapheme_lengths)
        tokens_phoneme_lengths = torch.stack(tokens_phoneme_lengths)

        return new_audio_signal, audio_lengths, new_tokens_grapheme, tokens_grapheme_lengths, new_tokens_phoneme, tokens_phoneme_lengths


class SpeechLLMDataset(SpeechRecognitionDataset):
    """Extends SpeechRecognitionDataset to return LLM-tokenized prompt/response pairs."""

    def set_llm_tokenizer(
        self,
        llm_tokenizer: AutoTokenizer,
        pre_prompt: str = "",
        post_prompt: str = "Transcribe:",
    ):
        self.llm_tokenizer = llm_tokenizer
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        speech, _ = super().__getitem__(idx)
        transcript = self.dataset[idx]["text"]

        pre_ids = self.llm_tokenizer.encode(
            self.pre_prompt, add_special_tokens=True, return_tensors="pt"
        )[0]
        post_ids = self.llm_tokenizer.encode(
            self.post_prompt, add_special_tokens=False, return_tensors="pt"
        )[0]
        output_ids = self.llm_tokenizer.encode(
            transcript, add_special_tokens=False, return_tensors="pt"
        )[0]
        eos = torch.tensor([self.llm_tokenizer.eos_token_id], dtype=torch.long)
        output_ids = torch.cat([output_ids, eos])

        return speech, pre_ids, post_ids, output_ids

    def collate_llm_data(self, batch: List) -> Tuple[torch.Tensor, ...]:
        wavs = [b[0][0] for b in batch]
        wav_lengths = [torch.tensor(len(w)) for w in wavs]
        max_wav_len = max(wav_lengths).item()

        padded_wavs = []
        for sig, sig_len in zip(wavs, wav_lengths):
            if sig_len < max_wav_len:
                sig = nn.functional.pad(sig, (0, max_wav_len - sig_len))
            padded_wavs.append(sig)
        padded_wavs = torch.stack(padded_wavs)
        wav_lengths = torch.stack(wav_lengths)

        pad_id = self.llm_tokenizer.pad_token_id or 0
        pre_ids = _pad_sequence([b[1] for b in batch], pad_id)
        post_ids = _pad_sequence([b[2] for b in batch], pad_id)
        output_ids = _pad_sequence([b[3] for b in batch], -100)

        return padded_wavs, wav_lengths, pre_ids, post_ids, output_ids


def _pad_sequence(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return padded