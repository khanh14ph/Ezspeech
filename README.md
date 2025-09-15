# EzSpeech

Easy speech recognition toolkit built on PyTorch Lightning.

## Features

- CTC and TDT-based ASR models
- Multiple encoder architectures (Conformer, Fast Conformer)
- GPU-optimized inference
- Pre-trained model support

## Installation

```bash
git clone https://github.com/khanh14ph/EzSpeech.git
cd EzSpeech
pip install -e .
```

## Training

```bash
# Train CTC_SC model
python train.py --config-name=ctc_sc

# Train TDT model
python train.py --config-name=asr
```

## Dataset Format

JSONL format:
```json
{"audio_filepath": "/path/to/audio.wav", "text": "transcription text", "duration": 3.2}
```
