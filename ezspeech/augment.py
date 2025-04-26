import random
from typing import Sequence, Optional, Union, List

import librosa
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from lightspeech.utils.common import load_dataset
from lightspeech.utils.operation import fft_convolution


class SpeedPerturbation:
    r"""Adjust the speed of the input by that factor.

    Args:
        orig_freqs (int or Sequence[int]): original frequency of the signals
        in ``waveform``.
        factors (Sequence[float]): factors by which to adjust speed of input.
            Values greater than 1.0 compress ``waveform`` in time,
            whereas values less than 1.0 stretch ``waveform`` in time.
    """

    def __init__(
        self, orig_freqs: Union[int, Sequence[int]], factors: Sequence[float]
    ) -> None:

        if isinstance(orig_freqs, int):
            orig_freqs = [orig_freqs]

        self.orig_freqs = orig_freqs

        self.transforms = {
            freq: T.SpeedPerturbation(freq, factors) for freq in orig_freqs
        }

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        r"""Adjust the speed of audio.

        Args:
            speech (Tensor): tensor of audio of dimension `(..., time)`.
            sample_rate (int): the sample rate of audio signal.

        Returns:
            Tensor
                speed-adjusted waveform, with shape `(..., new_time)`.
        """

        if sample_rate not in self.orig_freqs:
            raise ValueError(f"Sample rate {sample_rate} is not supported")

        speech, _ = self.transforms[sample_rate](speech)

        return speech


class Speed:
    r"""Adjust the speed of the input by that factor.

    Args:
        orig_freqs (int or Sequence[int]): original frequency of the signals
        in ``waveform``.
        factors (Sequence[float]): factors by which to adjust speed of input.
            Values greater than 1.0 compress ``waveform`` in time,
            whereas values less than 1.0 stretch ``waveform`` in time.
        probability (float): the probability of applying this augmentation.
    """

    def __init__(
        self,
        orig_freqs: Union[int, Sequence[int]],
        factors: Sequence[float],
        probability: float = 1.0,
    ) -> None:

        if isinstance(orig_freqs, int):
            orig_freqs = [orig_freqs]

        self.factor = factors
        self.orig_freqs = orig_freqs
        self.probability = probability

        self.transforms = {
            freq: {factor: T.Speed(freq, factor) for factor in factors}
            for freq in orig_freqs
        }  # noqa

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        r"""Adjust the speed of audio.

        Args:
            speech (Tensor): tensor of audio of dimension `(..., time)`.
            sample_rate (int): the sample rate of audio signal.

        Returns:
            Tensor
                speed-adjusted waveform, with shape `(..., new_time)`.
        """

        if random.random() > self.probability:
            return speech

        if sample_rate not in self.orig_freqs:
            raise ValueError(f"Sample rate {sample_rate} is not supported")

        factor = random.choice(self.factor)
        speech, _ = self.transforms[sample_rate][factor](speech)

        return speech


class TrimAudioSample(object):
    def __init__(
        self,
        factor: float,
        min_length: float,
        max_length: float,
        probability: float,
    ):
        self.factor = factor
        self.min_length = min_length
        self.max_length = max_length
        self.probability = probability

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech

        audio_length = speech.size(1) / sample_rate

        sample_length = self.factor * audio_length
        sample_length = min(self.max_length, sample_length)
        sample_length = max(self.min_length, sample_length)

        max_start_index = (audio_length - sample_length) * sample_rate
        start_index = random.randint(0, max(0, int(max_start_index)))

        length = int(sample_length * sample_rate)
        sample = speech[:, start_index : start_index + length]

        return sample


class ApplyImpulseResponse(object):
    def __init__(
        self,
        rir_filepath_8k: str = None,
        rir_filepath_16k: str = None,
        second_before_peak: Optional[float] = 0.01,
        second_after_peak: Optional[float] = 0.5,
        probability: Optional[float] = 0.2,
    ):
        self.probability = probability
        self.second_before_peak = second_before_peak
        self.second_after_peak = second_after_peak

        if rir_filepath_8k:
            self.rir_8k = load_dataset(rir_filepath_8k)
        if rir_filepath_16k:
            self.rir_16k = load_dataset(rir_filepath_16k)

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech

        if int(sample_rate) == 8000 and hasattr(self, "rir_8k"):
            rir_dataset = self.rir_8k
        elif int(sample_rate) == 16000 and hasattr(self, "rir_16k"):
            rir_dataset = self.rir_16k
        else:
            return speech

        rir_data = random.choice(rir_dataset)
        rir_filepath = rir_data["audio_filepath"]
        rir, sample_rate = torchaudio.load(rir_filepath)

        peak_index = rir.argmax()
        start_index = int(peak_index - self.second_before_peak * sample_rate)
        end_index = int(peak_index + self.second_after_peak * sample_rate)
        start_index = max(0, start_index)
        end_index = min(rir.size(1), end_index)

        rir = rir[:, start_index:end_index]
        rir /= rir.norm() + 1e-9
        rir = torch.flip(rir, [1])
        rir = rir[None, ...]

        padded_speech = F.pad(speech, (rir.size(2) - 1, 0))
        padded_speech = padded_speech[None, ...]

        reverbed_speech = fft_convolution(padded_speech, rir)[0]
        reverbed_speech *= speech.norm() / (reverbed_speech.norm() + 1e-9)
        reverbed_speech = reverbed_speech.clamp(-1.0, 1.0)

        return reverbed_speech


class AddBackgroundNoise(object):
    def __init__(
        self,
        noise_filepath_8k: str = None,
        noise_filepath_16k: str = None,
        min_snr_db: Optional[float] = 0.0,
        max_snr_db: Optional[float] = 30.0,
        probability: Optional[float] = 0.2,
    ):
        self.probability = probability
        self.snr_db = torch.distributions.Uniform(min_snr_db, max_snr_db)

        if noise_filepath_8k:
            self.noise_8k = load_dataset(noise_filepath_8k)
        if noise_filepath_16k:
            self.noise_16k = load_dataset(noise_filepath_16k)

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech

        if int(sample_rate) == 8000 and hasattr(self, "noise_8k"):
            noise_dataset = self.noise_8k
        elif int(sample_rate) == 16000 and hasattr(self, "noise_16k"):
            noise_dataset = self.noise_16k
        else:
            return speech

        noise_data = random.choice(noise_dataset)
        noise_filepath = noise_data["audio_filepath"]
        noise_duration = noise_data["duration"]

        speech_duration = speech.size(1) / sample_rate
        mismatch = int((noise_duration - speech_duration) * sample_rate)
        if mismatch > 0:
            frame_offset = random.randint(0, mismatch)
            noise, __ = torchaudio.load(
                noise_filepath,
                frame_offset=frame_offset,
                num_frames=speech.size(1),
            )
            rms_noise = noise.square().mean().sqrt() + 1e-9
        else:
            noise, __ = torchaudio.load(noise_filepath)
            rms_noise = noise.square().mean().sqrt() + 1e-9
            frame_offset = random.randint(0, -mismatch)
            noise = F.pad(noise, (frame_offset, -mismatch - frame_offset))

        snr_db = self.snr_db.sample()
        rms_speech = speech.square().mean().sqrt() + 1e-9
        scale = 10 ** (-snr_db / 20) * rms_speech / rms_noise

        noise = F.pad(noise, (0, speech.size(1) - noise.size(1)))
        noisy_speech = speech + scale * noise
        noisy_speech *= speech.norm() / (noisy_speech.norm() + 1e-9)
        noisy_speech = noisy_speech.clamp(-1.0, 1.0)

        return noisy_speech


# TODO move to device
class PitchShift(object):
    def __init__(
        self,
        min_step: int = -5,
        max_step: int = 5,
        sample_rates: List[int] = [8000, 16000, 22050],
        probability: float = 1.0,
    ) -> None:
        """Pitch shift

        Args:
            min_step (int, optional): Minimum number of steps to shift speech.
                Defaults to -5.
            max_step (int, optional): Maximum number of steps to shift speech.
                Defaults to 5.
            sample_rates (List[int], optional): List of sample rate.
                Defaults to [8000, 16000, 22050].
            probability (float, optional): Probability of applying forward
                speech. Defaults to 1.0.
        """
        self.min_step = min_step
        self.max_step = max_step
        self.probability = probability

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech
        n_steps = random.randint(self.min_step, self.max_step)
        speech = librosa.effects.pitch_shift(
            speech.numpy(), sr=sample_rate, n_steps=n_steps
        )
        return torch.tensor(speech)


class TimeMasking(object):
    def __init__(
        self,
        time_masks: Optional[int] = 10,
        time_width: Optional[float] = 0.05,
    ):
        self.time_masks = time_masks
        self.time_width = time_width
        self.augment = T.TimeMasking(1)

    def apply(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.unsqueeze(0)
        time_width = int(self.time_width * feature.size(-1))
        self.augment.mask_param = max(time_width, 1)
        for __ in range(self.time_masks):
            feature = self.augment(feature)
        return feature.squeeze(0)


class FrequencyMasking(object):
    def __init__(
        self,
        freq_masks: Optional[int] = 1,
        freq_width: Optional[int] = 27,
    ):
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.augment = T.FrequencyMasking(freq_width)

    def apply(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.unsqueeze(0)
        for __ in range(self.freq_masks):
            feature = self.augment(feature)
        return feature.squeeze(0)


class TimeStretch(object):

    def __init__(self, factors: Sequence[float], probability: float = 1.0) -> None:
        """Time stretch."""
        self.probability = probability
        self.factors = factors

        self.transforms = {
            factor: T.TimeStretch(n_freq=128, fixed_rate=factor) for factor in factors
        }

    def apply(self, feature: torch.Tensor) -> torch.Tensor:
        if random.random() > self.probability:
            return feature

        factor = random.choice(self.factors)
        feature = self.transforms[factor](feature)
        feature = torch.real(feature)
        return feature


class Spliceout(object):
    def __init__(self, time_mask_param: int, repeat: int) -> None:
        """Spliceout augmentation.

        Args:
            time_mask_param (int): Number of time masking.
            repeat (int): Number of repeatation.
        """
        self.mask_width = time_mask_param
        self.repeat = repeat

    def apply(self, spec: torch.Tensor) -> torch.Tensor:
        time_width = int(self.mask_width * spec.size(-1))
        D, T = spec.size()
        mask = torch.ones(T)

        for _ in range(self.repeat):
            start_idx = random.randint(0, T)
            block_width = random.randint(0, time_width)
            end_idx = min(start_idx + block_width, T)
            mask[start_idx:end_idx] = 0

        spec = spec[:, mask.bool()]
        return spec


class LengthPertubation(object):
    def __init__(
        self,
        drop_probability: float = 0.5,
        drop_width: float = 0.1,
        num_drop: int = 3,
        insert_probability: float = 0.5,
        insert_width: float = 0.1,
        num_inserts: int = 5,
    ) -> None:
        """Length Pertubation.

        Args:
            drop_probability (float): Drop feature probability.
            drop_width (float): Drop width.
            num_drop (int): Number of drop parts.
            insert_probability (float): Insert feature probability.
            insert_width (float): Insert width.
            num_inserts (int): Number of insertion.
        """
        self.ps = drop_probability
        self.rs = drop_width
        self.Ts = num_drop
        self.pp = insert_probability
        self.rp = insert_width
        self.Tp = num_inserts

    def apply(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.T
        tau = feature.shape[0]
        mask = torch.ones(tau, dtype=bool)
        time_width = int(self.rs * tau)

        # Drop frames
        if random.random() > self.ps:
            for _ in range(self.Ts):
                length = random.randint(0, time_width)
                start = random.randint(0, tau - length)
                mask[start : start + length] = False
            feature = feature[mask]

        # Insert frames
        if random.random() > self.pp:
            for _ in range(self.Tp):
                length = random.randint(0, time_width)
                start = random.randint(0, feature.shape[0])
                mask = torch.zeros(length, feature.shape[1])
                feature = torch.cat((feature[0:start, :], mask, feature[start:, :]), 0)

        return feature.T
