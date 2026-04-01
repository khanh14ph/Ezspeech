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
| CTC-LLM | `config/ctc_llm.yaml` | `ASR_ctc_llm_training` |

Both use a Conformer encoder.

## Dataset format

JSONL, one sample per line:

```json
{"audio_filepath": "relative/or/absolute.wav", "text": "transcript", "duration": 3.2}
```

## Training

```bash
# CTC
python scripts/train.py --config-name=ctc

# CTC-LLM
python scripts/train.py --config-name=ctc_llm
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
from ezspeech.models.ctc import ASR_ctc_inference

model = ASR_ctc_inference(
    filepath="my_model.ckpt",
    device="cuda",
    tokenizer_path="/path/to/tokenizer.model",
)

# CTC greedy
texts = model.transcribe(["audio1.wav", "audio2.wav"])
```

## Project structure

```
config/          # Hydra configs (ctc.yaml, ctc_llm.yaml)
scripts/         # train.py, train_tokenizer.py, ...
ezspeech/
  models/        # ASR_ctc_training, ASR_ctc_llm_training, ASR_ctc_inference
  modules/
    encoder/     # ConformerEncoder
    decoder/     # ConvASRDecoder
    losses/      # CTCLoss
    data/        # dataset, sampler, augmentation, tokenizer
  optims/        # NoamAnnealing scheduler
tokenizer/       # SentencePiece models
```
