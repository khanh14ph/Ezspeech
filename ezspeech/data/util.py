import torch
import torchaudio
import torchaudio.transforms as T

def extract_audio_feature(wav_form: torch.Tensor,sr: int):
    tranform= T.MelSpectrogram(
            sample_rate=sr,
            n_fft=int(0.05*sr),
            win_length=int(0.025*sr),
            hop_length=int(0.01*sr),
            center=False,
            n_mels=128,
        )
    res = tranform(wav_form)
    res = res.squeeze(0).clamp(1e-5).log()
    return res