from ezspeech.models.ASR import RNNT_CTC_Training
import torch

import os
a = RNNT_CTC_Training.load_from_checkpoint(
    "/home4/khanhnd/lightning_logs/oov/testcode2/checkpoints/last.ckpt"
)
os.makedirs("/home4/khanhnd/exported_checkpoint/models", exist_ok=True)
a.export_ez_checkpoint("/home4/khanhnd/exported_checkpoint/models")
