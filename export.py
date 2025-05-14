from ezspeech.tasks.recognition import SpeechRecognitionTask
import torch
a=SpeechRecognitionTask.load_from_checkpoint("/home4/khanhnd/lightning_logs/oov/testcode2/checkpoints/last.ckpt")
a.export_checkpoint("/home4/khanhnd/exported_checkpoint/hehe.ckpt")