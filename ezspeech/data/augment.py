import torch
import torchaudio
import torchaudio.transforms as T
from ezspeech.utils.common import load_dataset
import random
import torch.nn.functional as F

from typing import Optional
class time_masking:
    def __init__(self,time_width=0.05,time_masks=10,prob=0.2):
        self.time_width=time_width
        self.time_masks=time_masks
        self.augment=T.TimeMasking(1)
        self.prob=prob
    def __call__(self,feature):
        if random.uniform(0,1)< self.prob:
            
            feature=feature.unsqueeze(0)
            time_width=int(self.time_width*feature.size(-1))
            self.augment.mask_param=max(time_width,1)
            for _ in range(self.time_masks):
                
                feature=self.augment(feature)
            return feature.squeeze(0)
        else:
            return feature
class frequency_masking:
    def __init__(self,freq_width=27,freq_masks=1,prob=0.2):
        self.freq_width=freq_width
        self.freq_masks=freq_masks
        self.augment=T.FrequencyMasking(freq_width)
        self.prob=prob
    def __call__(self,feature):
        if random.uniform(0,1)< self.prob:
            feature=feature.unsqueeze(0)
            for _ in range(self.freq_masks):
                
                feature=self.augment(feature)
            return feature.squeeze(0)
        else:
            return feature

class AddBackgroundNoise(object):

    def __init__(

        self,

        noise_filepath_16k: str = None,

        min_snr_db: Optional[float] = 0.0,

        max_snr_db: Optional[float] = 30.0,

        probability: Optional[float] = 0.2,

    ):

        self.probability = probability

        self.snr_db = torch.distributions.Uniform(min_snr_db, max_snr_db)



        self.noise_dataset = load_dataset(noise_filepath_16k)



    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        sample_rate=16000

        if random.random() > self.probability:

            return speech

        noise_data = random.choice(self.noise_dataset)

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