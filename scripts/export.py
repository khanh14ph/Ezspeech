from ezspeech.models.tdt import ASR_tdt_training
model=ASR_tdt_training.load_from_checkpoint("/scratch/midway2/khanhnd/lightning_logs/tdt/version_4/checkpoints/epoch=7-val_wer=0.13092-step=146239.ckpt",weights_only=False)
model.export_checkpoint("../exported_checkpoint/ckpt_tdt.ckpt")