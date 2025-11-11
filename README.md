# EzSpeech

A modern, easy-to-use speech recognition toolkit built on PyTorch Lightning. EzSpeech provides state-of-the-art ASR models with clean APIs for training and evaluation.

## 🚀 Features

- **Multiple Model Architectures**: CTC and Transducer-based ASR models
- **Advanced Encoders**: Conformer, Fast Conformer architectures
- **Easy Training**: Simplified training workflows with Hydra configuration
- **Comprehensive Evaluation**: Detailed metrics and error analysis
- **GPU Optimization**: Efficient inference and training
- **Pre-trained Models**: Support for transfer learning

## 📦 Installation

### Quick Install

```bash
git clone https://github.com/khanh14ph/EzSpeech.git
cd EzSpeech
pip install -e .
```

### Development Install

```bash
git clone https://github.com/khanh14ph/EzSpeech.git
cd EzSpeech
pip install -e ".[dev]"
```

## 🚂 Training

### Quick Start

```bash
# Train CTC model with grapheme+phoneme
python scripts/train.py --config-name=ctc_sc

# Train standard CTC model
python scripts/train.py --config-name=ctc

# Train Transducer model
python scripts/train.py --config-name=asr
```

### Custom Configuration

Create your own config file in `config/` directory:

```yaml
# config/my_config.yaml
dataset:
  train_ds:
    filepaths:
      - /path/to/train.jsonl
    data_dir: /path/to/audio/
  val_ds:
    filepaths:
      - /path/to/val.jsonl
    data_dir: /path/to/audio/

model:
  d_model: 512
  vocab_size: 1024
  # ... other model parameters

trainer:
  max_epochs: 20
  devices: [0]
  precision: 16
```

Then train with:

```bash
python scripts/train.py --config-name=my_config
```

## 📊 Evaluation

### Evaluation Script

Use the evaluation utilities in `ezspeech/script/eval.py` for evaluating your models.

### Metrics

EzSpeech provides comprehensive metrics:

- **Word Error Rate (WER)** and **Character Error Rate (CER)**
- **Sentence-level accuracy**
- **Detailed error analysis** (substitutions, insertions, deletions)
- **Length statistics**

## 🎤 Inference

### Inference Script

Use the inference script for transcribing audio files:

```bash
python ezspeech/script/infer.py \
  --checkpoint /path/to/checkpoint.pt \
  --tokenizer /path/to/tokenizer.model \
  --input /path/to/audio.wav
```

For batch processing and detailed usage, refer to the script's help:

```bash
python ezspeech/script/infer.py --help
```

## 🚀 Deployment

### Docker Deployment

Test locally with Docker:

```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f
```

For detailed deployment options and configurations, see [DEPLOYMENT.md](DEPLOYMENT.md).

## 📁 Dataset Format

EzSpeech uses JSONL format for datasets:

```json
{"audio_filepath": "/path/to/audio.wav", "text": "transcription text", "duration": 3.2}
{"audio_filepath": "/path/to/audio2.wav", "text": "another transcription", "duration": 4.1}
```

### Required Fields

- `audio_filepath`: Path to audio file (relative to `data_dir` or absolute)
- `text`: Ground truth transcription
- `duration`: Audio duration in seconds (optional but recommended)

### Supported Audio Formats

- WAV, FLAC, MP3, OGG
- Sample rates: 8kHz, 16kHz, 22kHz, 44.1kHz (automatically resampled to 16kHz)
- Mono or stereo (automatically converted to mono)

## 🏗️ Project Structure

```
EzSpeech/
├── config/                 # Configuration files
│   ├── ctc_sc.yaml         # CTC with grapheme+phoneme
│   ├── ctc.yaml            # Standard CTC
│   ├── asr.yaml            # Transducer model
│   ├── streaming.yaml      # Streaming model
│   ├── eval.yaml           # Evaluation config
│   └── test.yaml           # Test configuration
├── scripts/                # Main scripts
│   ├── train.py            # Training script
│   ├── build_lexicon.py    # Build lexicon
│   ├── csv_to_jsonl.py     # Data conversion
│   └── export.py           # Model export
├── examples/               # Usage examples
│   ├── websocket_client.py # WebSocket client example
│   ├── evaluation_usage.md # Evaluation examples
│   └── README.md           # Examples documentation
├── ezspeech/              # Core package
│   ├── models/            # Model definitions
│   ├── modules/           # Lightning modules
│   ├── layers/            # Neural network layers
│   ├── script/            # Inference and utility scripts
│   │   ├── infer.py                  # Inference script
│   │   ├── eval.py                   # Evaluation utilities
│   │   ├── train_tokenizer.py        # Tokenizer training
│   │   └── validate_training.py      # Training validation
│   └── utils/             # Utilities
├── docs/                  # Documentation
├── demo/                  # Demo scripts
├── dockerfile             # Container image
└── docker-compose.yml     # Local development
```

## 🔧 Configuration

EzSpeech uses [Hydra](https://hydra.cc/) for configuration management. Key configuration sections:

### Dataset Configuration

```yaml
dataset:
  spe_file_grapheme: /path/to/grapheme.model    # SentencePiece model
  spe_file_phoneme: /path/to/phoneme.model      # Optional phoneme model
  train_ds:
    _target_: ezspeech.modules.data.dataset.SpeechRecognitionDataset
    filepaths: [/path/to/train.jsonl]
    data_dir: /path/to/audio/
  train_loader:
    max_batch_duration: 130  # Total audio seconds per batch
    num_bucket: 20           # Bucketing for efficiency
```

### Model Configuration

```yaml
model:
  d_model: 512              # Model dimension
  vocab_size: 1024          # Vocabulary size
  encoder:
    _target_: ezspeech.modules.encoder.conformer_offline.ConformerOfflineEncoder
    n_layers: 12
    d_model: 512
    ff_expansion_factor: 4
  ctc_decoder:
    _target_: ezspeech.modules.decoder.decoder.ConvASRDecoder
    num_classes: ${model.vocab_size}
```

### Training Configuration

```yaml
trainer:
  max_epochs: 20
  devices: [0]              # GPU devices
  precision: 16             # Mixed precision
  strategy: ddp             # Distributed training
  accumulate_grad_batches: 1
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ezspeech --cov-report=html
```

## 📈 Performance Tips

### Training Optimization

1. **Batch Size**: Use `max_batch_duration` instead of fixed batch size
2. **Mixed Precision**: Enable with `trainer.precision=16`
3. **Distributed Training**: Use `trainer.strategy=ddp` for multi-GPU
4. **Bucketing**: Use `num_bucket` for efficient batching

### Inference Optimization

1. **Batch Processing**: Process multiple files efficiently
2. **GPU Utilization**: Ensure optimal GPU usage during inference
3. **TorchScript**: Export models for faster inference
4. **ONNX**: Use ONNX runtime for deployment
5. **Quantization**: Apply post-training quantization

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🙏 Acknowledgments

- PyTorch Lightning team for the excellent framework
- Hydra team for configuration management
- The speech recognition research community

## 🎯 Quick Links

- **[📚 Online Documentation](https://khanh14ph.github.io/Ezspeech)** - Interactive guides and tutorials
- **[🐳 Deployment Options](DEPLOYMENT.md)** - Deployment methods and configurations
- **[💡 Examples](examples/)** - Code samples and usage examples

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/khanh14ph/EzSpeech/issues)
- **Discussions**: [GitHub Discussions](https://github.com/khanh14ph/EzSpeech/discussions)
- **Documentation**: [Online Docs](https://khanh14ph.github.io/Ezspeech)
