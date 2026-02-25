from ezspeech.models.ctc import ASR_ctc_training
model=ASR_ctc_training.load_from_checkpoint("/scratch/midway2/khanhnd/lightning_logs/icassp/version_1/checkpoints/last.ckpt",weights_only=False)
model.export_checkpoint("../exported_checkpoint/ckpt.ckpt")