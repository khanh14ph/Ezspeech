import time

from hydra.utils import instantiate
from omegaconf import OmegaConf

from ezspeech.utils.common import load_dataset
from jiwer import wer
BATCH_SIZE = 8

dataset = load_dataset(
    "/scratch/midway3/khanhnd/data/metadata/vivos_test.jsonl",
    data_dir="/scratch/midway3/khanhnd/data/audio",
)
audio_paths = [item["audio_filepath"] for item in dataset]
batches = [audio_paths[i : i + BATCH_SIZE] for i in range(0, len(audio_paths), BATCH_SIZE)]

cfg = OmegaConf.load("config/test/ctc.yaml")
model = instantiate(cfg.model)

# Warmup
model.run_encoder(batches[0])

decode_methods = [
    ("transcribe_ctc (CTC greedy)",    model.transcribe_ctc),
    ("transcribe_ctc_lm (CTC+LM)",     model.transcribe_ctc_lm),
]

references = [item["text"] for item in dataset]

for name, fn in decode_methods:
    total_time = 0.0
    hypotheses = []
    for batch in batches:
        t0 = time.perf_counter()
        hyps = fn(batch)
        total_time += time.perf_counter() - t0
        hypotheses.extend(hyps)
    avg = total_time / len(audio_paths)
    error_rate = wer(references, hypotheses)
    print(f"[{name}] total={total_time:.3f}s  avg/sample={avg:.3f}s  WER={error_rate:.4f}  ({len(audio_paths)} samples)")
