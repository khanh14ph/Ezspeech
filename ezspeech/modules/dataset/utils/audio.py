from typing import List, Tuple, Any, Optional

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torchaudio


def extract_filterbank(

    waveform: torch.Tensor,
    sample_rate: int,
    transformation=None,
    normalize: Optional[bool] = False,
) -> torch.Tensor:
    if transformation==None:
        transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(0.05 * sample_rate),
            win_length=int(0.025 * sample_rate),
            hop_length=int(0.01 * sample_rate),
            n_mels=128,
            center=False,
        )

    filterbank = transformation(waveform).squeeze(0)
    filterbank = filterbank.clamp(1e-5).log()

    return filterbank


def extract_melspectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
) -> torch.Tensor:
    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate, n_fft, win_length, hop_length, n_mels=n_mels
    )

    spectrogram = transformation(waveform).squeeze(0)
    spectrogram = spectrogram.clamp(1e-5).log()

    return spectrogram


def extract_spectrogram(
    waveform: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> torch.Tensor:
    transformation = torchaudio.transforms.Spectrogram(
        n_fft, win_length, hop_length, power=None
    )

    spectrogram = transformation(waveform)
    spectrogram = spectrogram.squeeze(0)

    return spectrogram


def inverse_spectrogram(
    spectrogram: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> torch.Tensor:
    transformation = torchaudio.transforms.InverseSpectrogram(
        n_fft, win_length, hop_length
    )

    waveform = transformation(spectrogram)
    waveform = waveform.unsqueeze(0)

    return waveform


def get_augmentation(config: DictConfig) -> Tuple[List[Any], List[Any]]:
    # audio augmentation
    augment_config = config.get("audio_augment", {})
    audio_augments = [instantiate(cfg) for cfg in augment_config.values()]

    # feature augmentation
    augment_config = config.get("feature_augment", {})
    feature_augments = [instantiate(cfg) for cfg in augment_config.values()]

    return audio_augments, feature_augments


