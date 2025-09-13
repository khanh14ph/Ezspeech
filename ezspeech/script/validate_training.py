from hydra.utils import instantiate
from omegaconf import OmegaConf

config = OmegaConf.load("/home4/khanhnd/Ezspeech/config/asr.yaml")
model = instantiate(config.model)
