from ezspeech.task.asr import ASR_ctc_task
a=ASR_ctc_task.load_from_checkpoint("/home4/khanhnd/Ezspeech/log/lightning_logs/test_code1/checkpoints/last.ckpt")
a.export_checkpoint("/home4/khanhnd/Ezspeech/exported_checkpoint/hehe.ckpt")