from ezspeech.task.asr import ASR_ctc_task
a=ASR_ctc_task.load_from_checkpoint("/home4/khanhnd/Ezspeech/log/self-condition/english/checkpoints/last.ckpt")
a.export_checkpoint("/home4/khanhnd/Ezspeech/exported_checkpoint/hehe.ckpt")