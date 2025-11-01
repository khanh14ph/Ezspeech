from ezspeech.models.ctc import ASR_ctc_training
model=ASR_ctc_training.load_from_checkpoint("/home3/khanhnd/lightning_logs/icassp/ctc/checkpoints/last.ckpt")
model.export_checkpoint("/home3/khanhnd/exported_checkpoint/ckpt.ckpt")