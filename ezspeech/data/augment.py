import torch
import torchaudio
import torchaudio.transforms as T
class time_masking:
    def __init__(self,proportion):
        self.proportion=proportion
        self.augment=T.TimeMasking(1)
    def __call__(self,feature):
        feature=feature.unsqueeze(0)
        time_width=int(self.proportion*feature.size(-1))
        self.augment.mask_param=max(time_width,1)
        feature=self.augment(feature)
        return feature.squeeze(0)
class frequency_masking:
    def __init__(self,proportion):
        self.proportion=proportion
        self.augment=T.FrequencyMasking(1)
    def __call__(self,feature):
        feature=feature.unsqueeze(0)
        time_width=int(self.proportion*feature.size(-1))
        self.augment.mask_param=max(time_width,1)
        feature=self.augment(feature)
        return feature.squeeze(0)