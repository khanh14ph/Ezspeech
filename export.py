import os

import torch

from ezspeech.models.ctc_recognition import ASR_ctc_training

a = ASR_ctc_training.load_from_checkpoint(
    "/home4/vuhl/last.ckpt"
)
a.export_checkpoint("/home3/khanhnd/exported_checkpoint/vi_ipa.ckpt")
