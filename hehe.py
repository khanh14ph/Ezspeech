from ezspeech.models.ASR import RNNT_CTC_Inference
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import torchaudio
config = OmegaConf.load("config/test/test.yaml")
Ez_model = instantiate(config.model)

audio_lst=[ "/home4/tuannd/vbee-asr/self-condition-asr/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/train-clean-100/374/180298/374-180298-0004.flac"]

audio = [audio1, audio2]
sr = [sr1, sr2]
enc, enc_length = Ez_model.forward_encoder(audio, sr)
print(Ez_model.greedy_ctc_decode(enc, enc_length))
print(Ez_model.greedy_tdt_decode(enc, enc_length))

