from hydra.utils import instantiate

from omegaconf import OmegaConf
cfg = OmegaConf.load("config/test/test.yaml")

model = instantiate(cfg.model)
audio_paths = ["/scratch/midway3/khanhnd/data/audio/youtube/audio_00009.flac"]
res=model.transcribe_lm(audio_paths)
print(res)