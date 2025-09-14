import os

import torch

from ezspeech.models.ctc_recognition import ASR_ctc_training

a = ASR_ctc_training.load_from_checkpoint(
    "/home4/khanhnd/lightning_logs/icassp/ctc/checkpoints/epoch=0-val_wer=1.00000-step=15.ckpt"
)
os.makedirs("/home4/khanhnd/exported_checkpoint/models", exist_ok=True)
a.export_checkpoint("/home4/khanhnd/exported_checkpoint/model.ckpt")
