from ezspeech.tasks.recognition import SpeechRecognitionTask
a=SpeechRecognitionTask.load_from_checkpoint("/home4/khanhnd/lightning_logs/oov/testcode1/checkpoints/epoch=2-val_ctc_loss=0.61033-step=7285.ckpt")
a.export_checkpoint("/home4/khanhnd/Ezspeech/exported_checkpoint/hehe.ckpt")