# EzSpeech

A lightweight Vietnamese ASR toolkit built on PyTorch Lightning and Hydra.

## Installation

```bash
git clone https://github.com/khanh14ph/EzSpeech.git
cd EzSpeech
pip install -e .
```

## Models

| Model | Config | Class |
|---|---|---|
| CTC | `config/ctc.yaml` | `ASR_ctc_training` |
| Hybrid CTC-TDT | `config/tdt.yaml` | `ASR_tdt_training` |

Both use a Conformer encoder. TDT additionally has a prediction network, joint network, and TDT loss alongside the CTC branch.

## Dataset format

JSONL, one sample per line:

```json
{"audio_filepath": "relative/or/absolute.wav", "text": "transcript", "duration": 3.2}
```

## Training

```bash
# CTC
python scripts/train.py --config-name=ctc

# Hybrid CTC-TDT
python scripts/train.py --config-name=tdt
```

Key config fields to set before training:

```yaml
dataset:
  spe_file: /path/to/tokenizer.model
  train_ds:
    filepaths: [/path/to/train.jsonl]
    data_dir: /path/to/audio/
  val_ds:
    filepaths: [/path/to/val.jsonl]
    data_dir: /path/to/audio/

trainer:
  devices: [0, 1]   # GPU indices
```

## Inference

Export a checkpoint from a trained model first:

```python
model.export_checkpoint("my_model.ckpt")
```

Then load and transcribe:

```python
from ezspeech.models.tdt import ASR_tdt_inference

model = ASR_tdt_inference(
    filepath="my_model.ckpt",
    device="cuda",
    tokenizer_path="/path/to/tokenizer.model",
)

# TDT greedy (fastest)
texts = model.transcribe(["audio1.wav", "audio2.wav"])

# TDT beam search
texts = model.transcribe_beam(["audio.wav"], beam_size=5, search_type="maes")

# TDT beam search + n-gram LM
texts = model.transcribe_beam(["audio.wav"], beam_size=5, search_type="maes",
                               ngram_lm_model="/path/to/lm.bin", ngram_lm_alpha=0.3)

# CTC greedy
texts = model.transcribe_ctc(["audio.wav"])
```

## Project structure

```
config/          # Hydra configs (ctc.yaml, tdt.yaml)
scripts/         # train.py, train_tokenizer.py, ...
ezspeech/
  models/        # ASR_ctc_training, ASR_tdt_training, ASR_tdt_inference
  modules/
    encoder/     # ConformerEncoder
    decoder/     # ConvASRDecoder, RNNTDecoder, RNNTJoint
    losses/      # CTCLoss, TDTLoss
    searcher/    # GreedyTDTInfer, BeamTDTInfer
    data/        # dataset, sampler, augmentation, tokenizer
  optims/        # NoamAnnealing scheduler
tokenizer/       # SentencePiece models
```
