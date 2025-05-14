import random
from typing import Sequence, Optional, Union, List

import librosa
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from numba import cuda


MAX_THREAD_BUFFER = 512
@cuda.jit()
def spec_augment_kernel(
    x: torch.Tensor,
    x_len: torch.Tensor,
    freq_starts: torch.Tensor,
    freq_widths: torch.Tensor,
    time_starts: torch.Tensor,
    time_widths: torch.Tensor,
    mask_value: float,
):
    """
    Numba CUDA kernel to perform SpecAugment in-place on the GPU.
    Parallelize over freq and time axis, parallel threads over batch.
    Sequential over masks (adaptive in time).

    Args:
        x: Pytorch tensor of shape [B, F, T] with the acoustic features.
        x_len: Pytorch tensor of shape [B] with the lengths of the padded sequence.
        freq_starts: Pytorch tensor of shape [B, M_f] with the start indices of freq masks.
        freq_widths: Pytorch tensor of shape [B, M_f] with the width of freq masks.
        time_starts: Pytorch tensor of shape [B, M_t] with the start indices of time masks.
        time_widths: Pytorch tensor of shape [B, M_t] with the width of time masks.
        mask_value: Float value that will be used as mask value.
    """
    f = cuda.blockIdx.x  # indexes the Freq dim
    t = cuda.blockIdx.y  # indexes the Time dim
    tid = cuda.threadIdx.x  # index of the current mask
    threads_per_block = cuda.blockDim.x

    # Compute the number of masks over freq axis
    len_f = freq_starts.shape[1]
    # For all samples in the batch, apply the freq mask
    for bidx in range(0, x.shape[0], threads_per_block):
        # Resolve the index of the batch (case where more masks than MAX_THREAD_BUFFER)
        bm_idx = bidx + tid

        # Access mask only if valid sample id in batch
        if bm_idx < x.shape[0]:
            # For `len_f` number of freq masks that must be applied
            for fidx in range(0, len_f):
                # Access the start index and width of this freq mask
                f_start = freq_starts[bm_idx, fidx]
                f_width = freq_widths[bm_idx, fidx]

                # If block idx `f` >= start and < (start + width) of this freq mask
                if f >= f_start and f < (f_start + f_width):
                    x[bm_idx, f, t] = mask_value

    # Compute the number of masks over time axis
    len_t = time_starts.shape[1]
    # For all samples in the batch, apply the time mask
    for b_idx in range(0, x.shape[0], threads_per_block):
        # Resolve the index of the batch (case where more masks than MAX_THREAD_BUFFER)
        bm_idx = b_idx + tid

        # Access mask only if valid sample id in batch
        if bm_idx < x.shape[0]:
            # For `len_t` number of freq masks that must be applied
            for tidx in range(0, len_t):
                # Access the start index and width of this time mask
                t_start = time_starts[bm_idx, tidx]
                t_width = time_widths[bm_idx, tidx]

                # If block idx `t` >= start and < (start + width) of this time mask
                if t >= t_start and t < (t_start + t_width):
                    # Current block idx `t` < current seq length x_len[b]
                    # This ensure that we mask only upto the length of that sample
                    # Everything after that index is padded value so unnecessary to mask
                    if t < x_len[bm_idx]:
                        x[bm_idx, f, t] = mask_value

def launch_spec_augment_kernel(
    x: torch.Tensor,
    x_len: torch.Tensor,
    freq_starts: torch.Tensor,
    freq_lengths: torch.Tensor,
    time_starts: torch.Tensor,
    time_lengths: torch.Tensor,
    freq_masks: int,
    time_masks: int,
    mask_value: float,
):
    """
    Helper method to launch the SpecAugment kernel

    Args:
        x: Pytorch tensor of shape [B, F, T] with the acoustic features.
        x_len: Pytorch tensor of shape [B] with the lengths of the padded sequence.
        freq_starts: Pytorch tensor of shape [B, M_f] with the start indices of freq masks.
        freq_widths: Pytorch tensor of shape [B, M_f] with the width of freq masks.
        time_starts: Pytorch tensor of shape [B, M_t] with the start indices of time masks.
        time_widths: Pytorch tensor of shape [B, M_t] with the width of time masks.
        freq_masks: Int value that determines the number of time masks.
        time_masks: Int value that determines the number of freq masks.
        mask_value: Float value that will be used as mask value.

    Returns:
        The spec augmented tensor 'x'
    """
    # Setup CUDA stream
    sh = x.shape
    stream = cuda.external_stream(torch.cuda.current_stream(x.device).cuda_stream)

    if time_masks > 0 or freq_masks > 0:
        # Parallelize over freq and time axis, parallel threads over batch
        # Sequential over masks (adaptive in time).
        blocks_per_grid = tuple([sh[1], sh[2]])
        # threads_per_block = min(MAX_THREAD_BUFFER, max(freq_masks, time_masks))
        threads_per_block = min(MAX_THREAD_BUFFER, x.shape[0])

        # Numba does not support fp16, force cast to fp32 temporarily at the expense of memory
        original_dtype = x.dtype
        cast_x = False
        if x.dtype == torch.float16:
            x = x.float()
            cast_x = True

        # Launch CUDA kernel
        spec_augment_kernel[blocks_per_grid, threads_per_block, stream, 0](
            x, x_len, freq_starts, freq_lengths, time_starts, time_lengths, mask_value
        )
        torch.cuda.synchronize()

        # Recast back to original dtype if earlier cast was performed
        if cast_x:
            x = x.to(dtype=original_dtype)

    return x

class SpecAugmentNumba(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    Utilizes a Numba CUDA kernel to perform inplace edit of the input without loops.
    Parallelize over freq and time axis, parallel threads over batch.
    Sequential over masks (adaptive in time).

    Args:
        freq_masks - how many frequency segments should be cut
        time_masks - how many time segments should be cut
        freq_width - maximum number of frequencies to be cut in one segment
        time_width - maximum number of time steps to be cut in one segment.
            Can be a positive integer or a float value in the range [0, 1].
            If positive integer value, defines maximum number of time steps
            to be cut in one segment.
            If a float value, defines maximum percentage of timesteps that
            are cut adaptively.
        rng: Ignored.
    """



    def __init__(
        self, freq_masks=0, time_masks=0, freq_width=10, time_width=0.1, mask_value=0.0,
    ):
        super().__init__()
        # Message to mention that numba specaugment kernel will be available
        # if input device is CUDA and lengths are provided

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.mask_value = mask_value


        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError('If `time_width` is a float value, must be in range [0, 1]')

            self.adaptive_temporal_width = True

    @torch.no_grad()
    def forward(self, input_spec, length):
        sh = input_spec.shape
        bs = sh[0]
        input_spec=input_spec.transpose(1,2)
        # Construct the freq and time masks as well as start positions
        if self.freq_masks > 0:
            freq_starts = torch.randint(
                0, sh[1] - self.freq_width + 1, size=[bs, self.freq_masks], device=input_spec.device
            )
            freq_lengths = torch.randint(0, self.freq_width + 1, size=[bs, self.freq_masks], device=input_spec.device)
        else:
            freq_starts = torch.zeros([bs, 1], dtype=torch.int64, device=input_spec.device)
            freq_lengths = torch.zeros([bs, 1], dtype=torch.int64, device=input_spec.device)

        if self.time_masks > 0:
            if self.adaptive_temporal_width:
                time_width = (length * self.time_width).int().clamp(min=1)
            else:
                time_width = (
                    torch.tensor(self.time_width, dtype=torch.int32, device=input_spec.device)
                    .unsqueeze(0)
                    .repeat(sh[0])
                )

            time_starts = []
            time_lengths = []
            for idx in range(sh[0]):
                time_starts.append(
                    torch.randint(
                        0, max(1, length[idx] - time_width[idx]), size=[1, self.time_masks], device=input_spec.device
                    )
                )
                time_lengths.append(
                    torch.randint(0, time_width[idx] + 1, size=[1, self.time_masks], device=input_spec.device)
                )

            time_starts = torch.cat(time_starts, 0)
            time_lengths = torch.cat(time_lengths, 0)

        else:
            time_starts = torch.zeros([bs, 1], dtype=torch.int64, device=input_spec.device)
            time_lengths = torch.zeros([bs, 1], dtype=torch.int64, device=input_spec.device)

        x = launch_spec_augment_kernel(
            input_spec,
            length,
            freq_starts=freq_starts,
            freq_lengths=freq_lengths,
            time_starts=time_starts,
            time_lengths=time_lengths,
            freq_masks=self.freq_masks,
            time_masks=self.time_masks,
            mask_value=self.mask_value,
        )
        x=x.transpose(1,2)
        return x
    
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