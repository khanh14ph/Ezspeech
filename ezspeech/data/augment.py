import torch
import torchaudio
import torchaudio.transforms as T
import random

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
    def __init__(self,time_width=0.05,time_masks=10,prob=0.2):
        self.time_width=time_width
        self.time_masks=time_masks
        self.augment=T.FrequencyMasking(1)
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
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class wav_augment:
    def __init__(self,prob):
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=prob),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=prob),
            PitchShift(min_semitones=-4, max_semitones=4, p=prob),
            Shift(p=prob),
        ])
    def __call__(self,wav):
        augment_wav=self.augment(samples=wav.numpy(), sample_rate=16000)
        return torch.tensor(augment_wav)
# Generate 2 seconds of dummy audio for the sake of example
# samples,_ =torchaudio.load("/home4/khanhnd/vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R106.wav")
# print("samples",samples.shape)
# # Augment/transform/perturb the audio data
# augmented_samples = augment(samples=samples.numpy(), sample_rate=16000)
# print("augmented_samples",augmented_samples.shape)
# torchaudio.save("hehe.wav",torch.tensor(augmented_samples),sample_rate=16000)