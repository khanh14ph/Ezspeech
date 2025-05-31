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
from ezspeech.modules.dataset.utils.text import Tokenizer
vn_word_lst=open("/home4/khanhnd/Ezspeech/ezspeech/resource/vn_word_lst.txt").read().splitlines()
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
        spe_file=None,
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechRecognitionDataset, self).__init__()
        sample_rate = 16000
        self.tokenizer=Tokenizer(vocab_file,spe_file=spe_file)
        self.vocab = open(vocab_file,encoding="utf-8").read().splitlines()
        self.vocab = [i.split()[0] for i in self.vocab ]

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

        tokens=self.tokenizer.encode(transcript)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return speech, tokens

    def __len__(self) -> int:
        return len(self.dataset)

class ASRDatasetBilingual(Dataset):
    def __init__(
        self,
        filepaths,
        vocab_file,
        spe_file=None,
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechRecognitionDataset, self).__init__()
        sample_rate = 16000
        self.tokenizer=Tokenizer(vocab_file,spe_file=spe_file)
        self.tokenizer_extra=Tokenizer("/home4/khanhnd/Ezspeech/ezspeech/resource/tokenizer/vi/vocab.txt",spe_file="/home4/khanhnd/Ezspeech/ezspeech/resource/tokenizer/vi/tokenizer.model")
        self.blank_idx=1024
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

        tokens=self.tokenize_bilingual(transcript,data["lan_id"])
        
        return speech, tokens
    def tokenize_mix_word(self,word):
        if word in vn_word_lst:
            tokens=self.tokenizer_extra.encode(word)
            tokens = torch.tensor(tokens, dtype=torch.long)+1025
        else:
            tokens=self.tokenizer.encode(word)
            tokens = torch.tensor(tokens, dtype=torch.long)
    def tokenize_bilingual(self,sentence,lan_id):

        if lan_id=="vi":
            tokens=self.tokenizer_extra.encode(sentence)
            tokens = torch.tensor(tokens, dtype=torch.long)+1025
            return tokens
        elif lan_id=="en":
            tokens=self.tokenizer.encode(sentence)
            tokens = torch.tensor(tokens, dtype=torch.long)
            return tokens
        else:
            temp=[]
            sentence=sentence.split(sentence)
            for i in sentence:
                temp.append(self.tokenize_mix_word(i))
            return torch.cat(temp)
    def __len__(self) -> int:
        return len(self.dataset)