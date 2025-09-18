# EzSpeech

A modern, easy-to-use speech recognition toolkit built on PyTorch Lightning. EzSpeech provides state-of-the-art ASR models with clean APIs for training, evaluation, and deployment.

## 🚀 Features

- **Multiple Model Architectures**: CTC and Transducer-based ASR models
- **Advanced Encoders**: Conformer, Fast Conformer architectures
- **Easy Training**: Simplified training workflows with Hydra configuration
- **Comprehensive Evaluation**: Detailed metrics and error analysis
- **Real-time Inference**: WebSocket server for live ASR
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

### Dependencies

Install additional dependencies for WebSocket server:

```bash
pip install websockets pyaudio
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

### Using Test Configuration

```bash
# Use the pre-configured test setup
python scripts/evaluate.py --config-name=test
```

### Custom Evaluation Configuration

Create your own evaluation config (copy from `config/eval.yaml`):

```bash
# Copy and modify the template
cp config/eval.yaml config/my_eval.yaml
# Edit config/my_eval.yaml with your paths
python scripts/evaluate.py --config-name=my_eval
```

### Configuration Override

```bash
# Override specific values from command line
python scripts/evaluate.py --config-name=test \
  model_checkpoint=/path/to/different/model.ckpt \
  output_dir=custom_results
```

### Metrics

EzSpeech provides comprehensive metrics:

- **Word Error Rate (WER)** and **Character Error Rate (CER)**
- **Sentence-level accuracy**
- **Detailed error analysis** (substitutions, insertions, deletions)
- **Length statistics**
- **Confidence-based metrics** (when available)

## 🌐 Real-time Inference (WebSocket)

### Start WebSocket Server

```bash
python scripts/serve_websocket.py \
  --model-path /path/to/model.ckpt \
  --host 0.0.0.0 \
  --port 8765
```

### Client Examples

See `examples/` directory for detailed client implementations.

#### File Transcription

```python
import asyncio
from examples.websocket_client import ASRWebSocketClient

async def transcribe_file():
    client = ASRWebSocketClient("ws://localhost:8765")
    await client.connect()

    result = await client.transcribe_file("/path/to/audio.wav")
    print(f"Transcription: {result}")

    await client.disconnect()

asyncio.run(transcribe_file())
```

#### Real-time Microphone

```python
async def real_time_demo():
    client = ASRWebSocketClient("ws://localhost:8765")
    await client.connect()

    # Stream from microphone for 10 seconds
    await client.stream_audio_from_microphone(duration_seconds=10.0)

    await client.disconnect()

asyncio.run(real_time_demo())
```

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
│   └── eval.yaml           # Evaluation config
├── scripts/                # Main scripts
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── serve_websocket.py  # WebSocket server
├── examples/               # Usage examples
│   ├── websocket_client.py # WebSocket client
│   └── README.md           # Examples documentation
├── ezspeech/              # Core package
│   ├── models/            # Model definitions
│   ├── modules/           # Lightning modules
│   ├── layers/            # Neural network layers
│   └── utils/             # Utilities
└── tests/                 # Unit tests
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

1. **TorchScript**: Export models for faster inference
2. **ONNX**: Use ONNX runtime for deployment
3. **Quantization**: Apply post-training quantization
4. **Batch Processing**: Process multiple files together

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch Lightning team for the excellent framework
- Hydra team for configuration management
- The speech recognition research community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/khanh14ph/EzSpeech/issues)
- **Discussions**: [GitHub Discussions](https://github.com/khanh14ph/EzSpeech/discussions)
- **Documentation**: [Wiki](https://github.com/khanh14ph/EzSpeech/wiki)
