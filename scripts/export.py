from ezspeech.models.ctc_llm import ASR_ctc_llm_training
model=ASR_ctc_llm_training.load_from_checkpoint("/scratch/midway2/khanhnd/lightning_logs/ctc_llm/version_2/checkpoints/last.ckpt",weights_only=False)
model.export_checkpoint("../exported_checkpoint/ckpt_ctc_llm.ckpt")