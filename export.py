from ezspeech.models.ASR import SpeechRecognitionTask
import torch

a = SpeechRecognitionTask.load_from_checkpoint(
    "/home4/khanhnd/lightning_logs/oov/testcode2/checkpoints/last.ckpt"
)
a.export_ez_checkpoint("/home4/khanhnd/models")
