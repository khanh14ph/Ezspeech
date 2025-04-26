import random
from collections import defaultdict
from typing import Tuple, List, Union, Optional
import json
from omegaconf import DictConfig
from tokenizers import Tokenizer

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm


from lightspeech.datas.text import (
    build_vocab,
    build_lexicon,
    tokenize,
    check_end_word,
)
from lightspeech.datas.audio import (
    get_augmentation,
    extract_filterbank,
    extract_melspectrogram,
)
from lightspeech.utils.common import load_dataset, time_reduction


def collate_ssl_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    feature_lengths = [len(f) for f in features]
    features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    targets = [b[1] for b in batch]
    targets = pad_sequence(targets, batch_first=True)

    return features, feature_lengths, targets


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
    audio_filepaths=[b[2] for b in batch]
    return features, feature_lengths, tokens, token_lengths,audio_filepaths


def collate_tts_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    token_idxs = [b[0] for b in batch]
    token_lens = [len(idxs) for idxs in token_idxs]
    token_idxs = pad_sequence(token_idxs, batch_first=True)
    token_lens = torch.tensor(token_lens, dtype=torch.long)

    dur_tgts = [b[1] for b in batch]
    dur_lens = [len(dur) for dur in dur_tgts]
    dur_tgts = pad_sequence(dur_tgts, batch_first=True)
    dur_lens = torch.tensor(dur_lens, dtype=torch.long)

    audio_tgts = [b[2] for b in batch]
    audio_lens = [len(audio) for audio in audio_tgts]
    audio_tgts = pad_sequence(audio_tgts, batch_first=True).transpose(1, 2)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)

    frame_refs = [b[3] for b in batch]
    frame_lens = [len(frame) for frame in frame_refs]
    frame_refs = pad_sequence(frame_refs, batch_first=True)
    frame_lens = torch.tensor(frame_lens, dtype=torch.long)

    return (
        token_idxs,
        token_lens,
        dur_tgts,
        dur_lens,
        audio_tgts,
        audio_lens,
        frame_refs,
        frame_lens,
    )


def collate_sc_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    feature_lengths = [len(f) for f in features]
    features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    targets = [b[1] for b in batch]
    targets = torch.stack(targets)

    return features, feature_lengths, targets


def collate_asv_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    lengths = [len(f) for f in features]
    anchor_features = pad_sequence(features, batch_first=True)
    anchor_lengths = torch.tensor(lengths, dtype=torch.long)

    features = [b[1] for b in batch]
    lengths = [len(f) for f in features]
    bonafide_features = pad_sequence(features, batch_first=True)
    bonafide_lengths = torch.tensor(lengths, dtype=torch.long)

    features = [b[2] for b in batch]
    lengths = [len(f) for f in features]
    spoofing_features = pad_sequence(features, batch_first=True)
    spoofing_lengths = torch.tensor(lengths, dtype=torch.long)

    speaker_identities = [b[3] for b in batch]
    speaker_identities = torch.stack(speaker_identities)

    return (
        anchor_features,
        anchor_lengths,
        bonafide_features,
        bonafide_lengths,
        spoofing_features,
        spoofing_lengths,
        speaker_identities,
    )


class SpeechRepresentationDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechRepresentationDataset, self).__init__()

        self.framerate = 4  # subsampling factor of acoustic encoder
        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        target = extract_filterbank(speech, sample_rate)

        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        target = target.t().unsqueeze(0)
        target_length = torch.tensor(target.size(1))
        target, __ = time_reduction(target, target_length, self.framerate)
        target = target.squeeze(0)

        return feature.t(), target

    def __len__(self) -> int:
        return len(self.dataset)


import os
import torchaudio.transforms as T

class SpeechRecognitionDataset(Dataset):
    def __init__(
        self,
        filepaths,
        vocab,
        bpe_model,
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechRecognitionDataset, self).__init__()

        self.lexicon = build_lexicon()
        sample_rate=16000
        self.transformation = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=int(0.05 * sample_rate),
                    win_length=int(0.025 * sample_rate),
                    hop_length=int(0.01 * sample_rate),
                    n_mels=128,
                    center=False,
                )
        self.vocab = open(vocab).read().splitlines()
        self.bpe_model=spm.SentencePieceProcessor(model_file=bpe_model)

        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        self.augmentation = augmentation
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation
        self.vietnamese_diacritics = {
            "a": ["á", "à", "ạ", "ã", "ả"],
            "ă": ["ắ", "ằ", "ẳ", "ẵ", "ẳ"],
            "â": ["ấ", "ầ", "ậ", "ẫ", "ẩ"],
            "e": ["é", "è", "ẹ", "ẽ", "ẻ"],
            "ê": ["ế", "ề", "ể", "ễ", "ể"],
            "i": ["í", "ì", "ị", "ĩ", "ỉ"],
            "o": ["ó", "ò", "ọ", "õ", "ỏ"],
            "ô": ["ố", "ồ", "ộ", "ỗ", "ổ"],
            "ơ": ["ớ", "ờ", "ợ", "ỡ", "ở"],
            "u": ["ú", "ù", "ụ", "ũ", "ủ"],
            "ư": ["ứ", "ừ", "ự", "ữ", "ử"],
            "y": ["ý", "ỳ", "ỵ", "ỹ", "ỷ"],
        }
        self.signed_char = [
            self.vietnamese_diacritics[j] for j in self.vietnamese_diacritics.keys()
        ]
        self.perpu_lst = [
            ["d", "r"],
            ("l", "n"),
            ("p", "b"),
            ("tr", "ch"),
            ("s", "x"),
        ]
        self.resampler = {8000:T.Resample(8000, 16000),24000: T.Resample(24000,16000)}
    def filter_long_audio(self, dataset):
        dataset1 = []
        for i in dataset:
            # if os.path.exists(i["audio_fielpath"]):
            if "duration" in i.keys():
                if i["duration"] < 10:
                    dataset1.append(i)
            else:
                dataset1.append(i)
        print("after remove >10 s", len(self.dataset))
        return dataset1

    def vowel_augment(self, x, prob):
        if random.random() > prob:
            return x
        else:
            res = []
            y = list(x)
            for i in y:
                replace_char = i
                for v in self.signed_char:
                    if i in v:
                        replace_char = random.choice(v)
                        break
                res.append(replace_char)
        final = "".join(res)
        if final in self.vocab:
            return final
        return x

    def consonant_augment(self, x, prob):
        if random.random() > prob:
            return x
        for i in self.perpu_lst:
            if x in i:
                return random.choice(i)
        return x

    def token_augment(self, token_lst, prob):
        res = []
        for i in token_lst:
            if check_end_word(i, self.vocab):
                res.append(self.vowel_augment(i, prob / 2))
                # res.append(i)
            else:
                res.append(self.consonant_augment(i, prob))
        return res

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        transcript = data["transcript"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        if sample_rate!=16000:
            if sample_rate not in self.resampler.keys():
                self.resampler[sample_rate]=T.Resample(sample_rate,16000)
            speech=self.resampler[sample_rate](speech)
            sample_rate=16000

        speech = speech.mean(dim=0, keepdim=True)

        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate,self.transformation)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        tokens = self.bpe_model.encode(transcript)
        # tokens = tokenize(transcript, self.vocab)

        # tokens = [self.vocab.index(token) for token in tokens]

        tokens = torch.tensor(tokens, dtype=torch.long)
        frame_length=feature.shape[1]/0.01
        if len(tokens)>frame_length:
            qweqwe
        return feature.t(), tokens, audio_filepath

    def __len__(self) -> int:
        return len(self.dataset)



# from transformers import AutoModel, AutoTokenizer

# tokenizer_bert = AutoTokenizer.from_pretrained(
#     "/data/khanhnd65/cache/hub/models--vinai--phobert-base-v2/snapshots/e2375d266bdf39c6e8e9a87af16a5da3190b0cc8"
# )

# import py_vncorenlp

# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=".")


def segment(x):

    output = rdrsegmenter.word_segment(x)
    return " ".join(output)


def collate_alignment_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    feature_lengths = [len(f) for f in features]
    features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    transcript_lst = [b[1] for b in batch]
    encodings = tokenizer_bert(
        transcript_lst, padding=True, truncation=True, return_tensors="pt"
    )

    tokens = encodings["input_ids"]
    tokens_attention_mask = encodings["attention_mask"]
    return features, feature_lengths, tokens, tokens_attention_mask


class AlignmentDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        vocab: Optional[str] = None,
        augmentation: Optional[DictConfig] = None,
        sample: int = None,
    ):
        super(AlignmentDataset, self).__init__()

        self.lexicon = build_lexicon()

        if vocab:
            self.vocab = open(vocab).read().splitlines()
        else:
            self.vocab = build_vocab()

        self.dataset = load_dataset(filepaths)[:sample]

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation
        self.vietnamese_diacritics = {
            "a": ["á", "à", "ạ", "ã", "ả"],
            "ă": ["ắ", "ằ", "ẳ", "ẵ", "ẳ"],
            "â": ["ấ", "ầ", "ậ", "ẫ", "ẩ"],
            "e": ["é", "è", "ẹ", "ẽ", "ẻ"],
            "ê": ["ế", "ề", "ể", "ễ", "ể"],
            "i": ["í", "ì", "ị", "ĩ", "ỉ"],
            "o": ["ó", "ò", "ọ", "õ", "ỏ"],
            "ô": ["ố", "ồ", "ộ", "ỗ", "ổ"],
            "ơ": ["ớ", "ờ", "ợ", "ỡ", "ở"],
            "u": ["ú", "ù", "ụ", "ũ", "ủ"],
            "ư": ["ứ", "ừ", "ự", "ữ", "ử"],
            "y": ["ý", "ỳ", "ỵ", "ỹ", "ỷ"],
        }
        self.signed_char = [
            self.vietnamese_diacritics[j] for j in self.vietnamese_diacritics.keys()
        ]
        self.perpu_lst = [
            ["d", "r"],
            ("l", "n"),
            ("p", "b"),
            ("tr", "ch"),
            ("s", "x"),
        ]

    def vowel_augment(self, x, prob):
        if random.random() > prob:
            return x
        else:
            res = []
            y = list(x)
            for i in y:
                replace_char = i
                for v in self.signed_char:
                    if i in v:
                        replace_char = random.choice(v)
                        break
                res.append(replace_char)
        final = "".join(res)
        if final in self.vocab:
            return final
        return x

    def consonant_augment(self, x, prob):
        if random.random() > prob:
            return x
        for i in self.perpu_lst:
            if x in i:
                return random.choice(i)
        return x

    def token_augment(self, token_lst, prob):
        res = []
        for i in token_lst:
            if check_end_word(i, self.vocab):
                # res.append(self.vowel_augment(i, prob))
                res.append(i)
            else:
                res.append(self.consonant_augment(i, prob))
        return res

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        transcript = data["transcript_segment"]
        speech, sample_rate = torchaudio.load(audio_filepath)
        speech = speech.mean(dim=0, keepdim=True)

        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        return feature.t(), transcript

    def __len__(self) -> int:
        return len(self.dataset)


class SpeechSynthesisDataset(Dataset):
    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechSynthesisDataset, self).__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.vocab = build_vocab()
        self.lexicon = build_lexicon()
        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        length = data["duration"]
        alignment = data["alignment"]

        clean_speech, sample_rate = torchaudio.load(audio_filepath)

        clean_feature = extract_melspectrogram(
            clean_speech,
            sample_rate,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.n_mels,
        )

        durations, tokens = [], []
        for token, start, end in alignment:
            duration = (end - start) / length * clean_feature.size(1)
            durations.append(max(1, int(duration)))

            token = self.vocab.index(token)
            tokens.append(token)

        mismatch = clean_feature.size(1) - sum(durations)
        bias = 1 if mismatch >= 0 else -1
        for _ in range(abs(mismatch)):
            if bias == -1:
                indices = [i for i, dur in enumerate(durations) if dur > 1]
            else:
                indices = [i for i in range(len(durations))]

            idx = random.choice(indices)
            durations[idx] += bias

        durations = torch.tensor(durations, dtype=torch.long)
        tokens = torch.tensor(tokens, dtype=torch.long)

        noisy_speech = clean_speech.clone()
        for augment in self.audio_augment:
            noisy_speech = augment.apply(noisy_speech, sample_rate)

        noisy_feature = extract_filterbank(noisy_speech, sample_rate)
        for augment in self.feature_augment:
            noisy_feature = augment.apply(noisy_feature)

        return tokens, durations, clean_speech.t(), noisy_feature.t()

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


class SpeakerRecognitionDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        speakers: str,
        augmentation: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.dataset = self._build_dataset(filepaths)
        self.speakers = self._load_speakers(speakers)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def _build_dataset(self, filepaths):
        dataset = load_dataset(filepaths)
        return dataset

    def _load_speakers(self, filepath):
        with open(filepath) as f:
            speakers = f.read().split("\n")[:-1]
        return speakers

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        filepath = data["audio_filepath"]
        speaker = data["speaker"]
        label = data.get("label", "bonafide")

        speech, sample_rate = torchaudio.load(filepath)

        duration = speech.size(1) / sample_rate
        if duration < 3.0:
            padding = int((3.0 - duration) / 2 * sample_rate)
            speech = torch.nn.functional.pad(speech, (padding, padding))

        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        speaker = speaker if label == "bonafide" else "__SPF__"
        speaker = self.speakers.index(speaker)
        speaker = torch.tensor(speaker, dtype=torch.long)

        return feature.t(), speaker

    def __len__(self) -> int:
        return len(self.dataset)


class SpeechVerificationDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        speakers: str,
        augmentation: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.dataset = self._build_dataset(filepaths)

        (
            self.bonafide_datastore,
            self.spoofing_datastore,
        ) = self._build_datastore(filepaths)

        self.speakers = self._load_speakers(speakers)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def _build_dataset(self, filepaths):
        dataset = load_dataset(filepaths)
        dataset = [data for data in dataset if data["label"] == "bonafide"]
        return dataset

    def _build_datastore(self, filepaths):
        dataset = load_dataset(filepaths)

        bonafide_datastore = defaultdict(list)
        spoofing_datastore = defaultdict(list)

        for data in dataset:
            speaker = data["speaker"]
            filepath = data["audio_filepath"]

            if data["label"] == "bonafide":
                bonafide_datastore[speaker].append(filepath)

            if data["label"] == "spoofing":
                spoofing_datastore[speaker].append(filepath)

        return bonafide_datastore, spoofing_datastore

    def _load_speakers(self, filepath):
        with open(filepath) as f:
            speakers = f.read().split("\n")[:-1]
        return speakers

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        anchor_data = self.dataset[idx]
        speaker = anchor_data["speaker"]

        anchor_filepath = anchor_data["audio_filepath"]
        anchor_feature = self._transform(anchor_filepath)

        bonafide_filepath = self._random_bonafide(speaker)
        bonafide_feature = self._transform(bonafide_filepath)

        spoofing_filepath = self._random_spoofing(speaker)
        spoofing_feature = self._transform(spoofing_filepath)

        speaker = self.speakers.index(speaker)
        speaker = torch.tensor(speaker, dtype=torch.long)

        return anchor_feature, bonafide_feature, spoofing_feature, speaker

    def _random_bonafide(self, speaker: str):
        sample = random.choice(
            [filepath for filepath in self.bonafide_datastore[speaker]]
        )
        return sample

    def _random_spoofing(self, speaker: str):
        if speaker in self.spoofing_datastore:
            sample = random.choice(
                [filepath for filepath in self.spoofing_datastore[speaker]]
            )

        else:
            speaker = random.choice(
                [spk for spk in self.spoofing_datastore if spk != speaker]
            )
            sample = random.choice(
                [filepath for filepath in self.spoofing_datastore[speaker]]
            )

        return sample

    def _gather_samples(self, target_speaker: str, target_label: str):
        samples = [
            filepath
            for (filepath, label) in self.datastore[target_speaker]
            if label == target_label
        ]
        return samples

    def _transform(self, audio_filepath: str) -> torch.Tensor:
        speech, sample_rate = torchaudio.load(audio_filepath)

        # duration = speech.size(1) / sample_rate
        # if duration < 3.0:
        #     padding = int((3.0 - duration) / 2 * sample_rate)
        #     speech = torch.nn.functional.pad(speech, (padding, padding))

        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        return feature.t()

    def __len__(self) -> int:
        return len(self.dataset)


class SpectrogramReconstructionDataset(Dataset):
    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.vocab = build_vocab()
        self.lexicon = build_lexicon()
        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def __getitem__(self, idx):
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        audio_length = data["duration"]
        alignment = data["alignment"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        feature = extract_melspectrogram(
            speech,
            sample_rate,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.n_mels,
        )

        durations, tokens = [], []
        for token, start, end in alignment:
            duration = (end - start) / audio_length * feature.size(1)
            durations.append(max(1, int(duration)))

            token = self.vocab.index(token)
            tokens.append(token)

        mismatch = feature.size(1) - sum(durations)
        bias = 1 if mismatch >= 0 else -1
        for _ in range(abs(mismatch)):
            if bias == -1:
                indices = [i for i, dur in enumerate(durations) if dur > 1]
            else:
                indices = [i for i in range(len(durations))]

            idx = random.choice(indices)
            durations[idx] += bias

        durations = torch.tensor(durations, dtype=torch.long)
        tokens = torch.tensor(tokens, dtype=torch.long)

        prompt_speech = speech.clone()
        for augment in self.audio_augment:
            prompt_speech = augment.apply(prompt_speech, sample_rate)

        return tokens, durations, prompt_speech.t(), feature.t()

    def __len__(self):
        return len(self.dataset)


class WaveformReconstructionDataset(Dataset):
    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.vocab = build_vocab()
        self.lexicon = build_lexicon()
        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def __getitem__(self, idx):
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_melspectrogram(
            speech,
            sample_rate,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.n_mels,
        )

        for augment in self.feature_augment:
            feature = augment.apply(feature)

        return feature.t(), speech.t()

    def __len__(self):
        return len(self.dataset)


def collate_mel_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    token_idxs = [b[0] for b in batch]
    token_lens = [len(idxs) for idxs in token_idxs]
    token_idxs = pad_sequence(token_idxs, batch_first=True)
    token_lens = torch.tensor(token_lens, dtype=torch.long)

    dur_tgts = [b[1] for b in batch]
    dur_lens = [len(dur) for dur in dur_tgts]
    dur_tgts = pad_sequence(dur_tgts, batch_first=True)
    dur_lens = torch.tensor(dur_lens, dtype=torch.long)

    prompt_wavs = [b[2] for b in batch]
    prompt_lens = [len(wav) for wav in prompt_wavs]
    prompt_wavs = pad_sequence(prompt_wavs, batch_first=True).transpose(1, 2)
    prompt_lens = torch.tensor(prompt_lens, dtype=torch.long)

    mel_tgts = [b[3] for b in batch]
    mel_lens = [len(mel) for mel in mel_tgts]
    mel_tgts = pad_sequence(mel_tgts, batch_first=True)
    mel_lens = torch.tensor(mel_lens, dtype=torch.long)

    return (
        token_idxs,
        token_lens,
        dur_tgts,
        dur_lens,
        prompt_wavs,
        prompt_lens,
        mel_tgts,
        mel_lens,
    )


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


if __name__ == "__main__":
    import py_vncorenlp

    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=".")

    text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

    output = rdrsegmenter.word_segment(text)
    print("".join(output))
