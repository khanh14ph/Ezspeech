from omegaconf import OmegaConf
from hydra.utils import instantiate
config= OmegaConf.load("/home4/khanhnd/Ezspeech/config/asr.yaml")
model = instantiate(config.model)