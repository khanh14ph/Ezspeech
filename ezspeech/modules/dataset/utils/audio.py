from typing import List, Tuple, Any, Optional

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch import nn
import torchaudio
from typing import Optional, Tuple, Union
# from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogram
def make_seq_mask_like(
    lengths: torch.Tensor, like: torch.Tensor, time_dim: int = -1, valid_ones: bool = True
) -> torch.Tensor:
    """

    Args:
        lengths: Tensor with shape [B] containing the sequence length of each batch element
        like: The mask will contain the same number of dimensions as this Tensor, and will have the same max
            length in the time dimension of this Tensor.
        time_dim: Time dimension of the `shape_tensor` and the resulting mask. Zero-based.
        valid_ones: If True, valid tokens will contain value `1` and padding will be `0`. Else, invert.

    Returns:
        A :class:`torch.Tensor` containing 1's and 0's for valid and invalid tokens, respectively, if `valid_ones`, else
        vice-versa. Mask will have the same number of dimensions as `like`. Batch and time dimensions will match
        the `like`. All other dimensions will be singletons. E.g., if `like.shape == [3, 4, 5]` and
        `time_dim == -1', mask will have shape `[3, 1, 5]`.
    """
    # Mask with shape [B, T]
    mask = torch.arange(like.shape[time_dim], device=like.device).repeat(lengths.shape[0], 1).lt(lengths.view(-1, 1))
    # [B, T] -> [B, *, T] where * is any number of singleton dimensions to expand to like tensor
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    # If needed, transpose time dim
    if time_dim != -1 and time_dim != mask.dim() - 1:
        mask = mask.transpose(-1, time_dim)
    # Maybe invert the padded vs. valid token values
    if not valid_ones:
        mask = ~mask
    return mask
class AudioToMelSpectrogramTA(nn.Module):
    def __init__(self,sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        use_torchaudio: bool = False,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if window_size:
            n_window_size = int(window_size * sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * sample_rate)
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self._sample_rate = sample_rate
        self._normalize_strategy = normalize
        self._use_log = log
        self._preemphasis_value = preemph
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value: Union[str, float] = log_zero_guard_value
        self.dither = dither
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.n_fft = n_fft
        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }
        self._mel_spec_extractor= torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=features,
            window_fn=self.torch_windows[window],
            mel_scale="slaney",
            norm=mel_norm,
            n_fft=n_fft,
            f_max=highfreq,
            f_min=lowfreq,
            wkwargs={"periodic": False},
        )
    
    def filter_banks(self):
        """Matches the analogous class"""
        return self._mel_spec_extractor.mel_scale.fb

    def _resolve_log_zero_guard_value(self, dtype: torch.dtype) -> float:
        if isinstance(self.log_zero_guard_value, float):
            return self.log_zero_guard_value
        return getattr(torch.finfo(dtype), self.log_zero_guard_value)

    def _apply_dithering(self, signals: torch.Tensor) -> torch.Tensor:
        if self.training and self.dither > 0.0:
            noise = torch.randn_like(signals) * self.dither
            signals = signals + noise
        return signals

    def _apply_preemphasis(self, signals: torch.Tensor) -> torch.Tensor:
        if self._preemphasis_value is not None:
            padded = torch.nn.functional.pad(signals, (1, 0))
            signals = signals - self._preemphasis_value * padded[:, :-1]
        return signals

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        out_lengths = input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()
        return out_lengths

    def _apply_pad_to(self, features: torch.Tensor) -> torch.Tensor:
        # Only apply during training; else need to capture dynamic shape for exported models
        if not self.training or self.pad_to == 0 or features.shape[-1] % self.pad_to == 0:
            return features
        pad_length = self.pad_to - (features.shape[-1] % self.pad_to)
        return torch.nn.functional.pad(features, pad=(0, pad_length), value=self.pad_value)

    def _apply_log(self, features: torch.Tensor) -> torch.Tensor:
        if self._use_log:
            zero_guard = self._resolve_log_zero_guard_value(features.dtype)
            if self.log_zero_guard_type == "add":
                features = features + zero_guard
            elif self.log_zero_guard_type == "clamp":
                features = features.clamp(min=zero_guard)
            else:
                raise ValueError(f"Unsupported log zero guard type: '{self.log_zero_guard_type}'")
            features = features.log()
        return features

    def _extract_spectrograms(self, signals: torch.Tensor) -> torch.Tensor:
        # Complex FFT needs to be done in single precision
        with torch.amp.autocast('cuda', enabled=False):
            features = self._mel_spec_extractor(waveform=signals)
        return features

    def _apply_normalization(self, features: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # For consistency, this function always does a masked fill even if not normalizing.
        mask: torch.Tensor = make_seq_mask_like(lengths=lengths, like=features, time_dim=-1, valid_ones=False)
        features = features.masked_fill(mask, 0.0)
        # Maybe don't normalize
        if self._normalize_strategy is None:
            return features
        # Use the log zero guard for the sqrt zero guard
        guard_value = self._resolve_log_zero_guard_value(features.dtype)
        if self._normalize_strategy == "per_feature" or self._normalize_strategy == "all_features":
            # 'all_features' reduces over each sample; 'per_feature' reduces over each channel
            reduce_dim = 2
            if self._normalize_strategy == "all_features":
                reduce_dim = [1, 2]
            # [B, D, T] -> [B, D, 1] or [B, 1, 1]
            means = features.sum(dim=reduce_dim, keepdim=True).div(lengths.view(-1, 1, 1))
            stds = (
                features.sub(means)
                .masked_fill(mask, 0.0)
                .pow(2.0)
                .sum(dim=reduce_dim, keepdim=True)  # [B, D, T] -> [B, D, 1] or [B, 1, 1]
                .div(lengths.view(-1, 1, 1) - 1)  # assume biased estimator
                .clamp(min=guard_value)  # avoid sqrt(0)
                .sqrt()
            )
            features = (features - means) / (stds + eps)
        else:
            # Deprecating constant std/mean
            raise ValueError(f"Unsupported norm type: '{self._normalize_strategy}")
        features = features.masked_fill(mask, 0.0)
        return features

    def forward(self, input_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_lengths = self._compute_output_lengths(input_lengths=length)
        signals = self._apply_dithering(signals=input_signal)
        signals = self._apply_preemphasis(signals=signals)
        features = self._extract_spectrograms(signals=signals)
        features = self._apply_log(features=features)
        features = self._apply_normalization(features=features, lengths=feature_lengths)
        features = self._apply_pad_to(features=features)
        features=features.transpose(1, 2)
        return features, feature_lengths



def get_augmentation(config: DictConfig) -> Tuple[List[Any], List[Any]]:
    # audio augmentation
    augment_config = config.get("audio_augment", {})
    audio_augments = [instantiate(cfg) for cfg in augment_config.values()]

    # feature augmentation
    augment_config = config.get("feature_augment", {})
    feature_augments = [instantiate(cfg) for cfg in augment_config.values()]

    return audio_augments, feature_augments
