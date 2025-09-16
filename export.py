import os

import torch

from ezspeech.models.ctc_recognition import ASR_ctc_training

a = ASR_ctc_training.load_from_checkpoint(
    "/home3/khanhnd/lightning_logs/icassp/ctc/checkpoints/last.ckpt"
)
a.export_checkpoint("/home3/khanhnd/exported_checkpoint/model.ckpt")
