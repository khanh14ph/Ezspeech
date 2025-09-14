# 🎤 EzSpeech - Easy Speech Recognition Toolkit

> A streamlined, customizable speech recognition toolkit built on NeMo foundations with PyTorch Lightning power, featuring all state-of-the-art ASR algorithms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-purple.svg)](https://lightning.ai)
[![NeMo](https://img.shields.io/badge/NeMo-Compatible-green.svg)](https://github.com/NVIDIA/NeMo)
[![CUDA](https://img.shields.io/badge/CUDA-Optimized-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Triton](https://img.shields.io/badge/Triton-Supported-FF6B35.svg)](https://triton-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 What Makes EzSpeech Special?

**EzSpeech** brings you the complete arsenal of NVIDIA NeMo's cutting-edge ASR algorithms in a simplified, customizable package. Built for researchers, developers, and ML engineers who demand both **performance** and **ease of use**.

- 🔥 **All NeMo SOTA algorithms** - TDT decode, Fast Conformer, GPU N-gram LM, and more
- ⚡ **CUDA-optimized inference** with CUDA Graph acceleration
- 🚀 **Triton kernel support** for maximum throughput
- 🎯 **Zero-hassle pretrained models** from NeMo ecosystem
- 🔧 **Easy customization** without complex codebases

## ✨ State-of-the-Art Features

### 🧠 **Advanced Architectures**
- **🚄 Fast Conformer**: Ultra-efficient attention with 8x speedup
- **🔄 Conformer**: Industry-standard transformer + convolution hybrid
- **⚡ Squeezeformer**: Optimized for mobile and edge deployment
- **🎯 ContextNet**: Streaming-optimized architecture

### 🎯 **Cutting-Edge Decoding**
- **🔥 TDT (Token-and-Duration Transducer)**: Next-gen streaming ASR
- **⚡ CUDA Graph TDT Decode**: Hardware-accelerated inference pipeline
- **🧠 GPU N-gram Language Model**: Lightning-fast LM scoring on GPU
- **🎪 Beam Search with LM**: Traditional beam search with language model fusion
- **🔄 Greedy Decode**: Fastest inference for real-time applications

### 🚀 **Performance Optimizations**
- **⚡ Triton Kernels**: Custom GPU kernels for maximum throughput
- **📈 CUDA Graphs**: Eliminate kernel launch overhead
- **🔧 Mixed Precision**: FP16/BF16 training and inference
- **🎯 Dynamic Batching**: Optimal GPU utilization
- **💾 Memory Optimization**: Gradient checkpointing and activation recomputation

## 🛠 Installation

```bash
# Install from PyPI (coming soon!)
pip install ezspeech

# Or install from source
git clone https://github.com/yourusername/EzSpeech.git
cd EzSpeech
pip install -e .
```

## 🏋️ Training

EzSpeech uses Hydra for configuration management and PyTorch Lightning for training. You can train models with different architectures using the provided configuration files.

### Quick Start Training

```bash
# Train a CTC model (default configuration)
python train.py

# Train with a specific config
python train.py --config-name=ctc
python train.py --config-name=asr     # TDT/RNN-T model
python train.py --config-name=streaming  # Streaming TDT model
```

### Available Configurations

- **`ctc.yaml`**: CTC-based ASR model with Conformer encoder
- **`asr.yaml`**: TDT (Token-and-Duration Transducer) model for non-streaming ASR
- **`streaming.yaml`**: Streaming TDT model for real-time applications

### Configuration Override

You can override any configuration parameter from the command line:

```bash
# Override training parameters
python train.py trainer.max_epochs=100 trainer.devices=[0,1]

# Override model parameters
python train.py model.encoder.n_layers=18 model.encoder.d_model=768

# Override dataset paths
python train.py dataset.train_ds.filepaths=[/path/to/train.jsonl] dataset.val_ds.filepaths=[/path/to/val.jsonl]

# Override optimizer settings
python train.py optimizer.lr=0.001 scheduler.warmup_steps=5000
```

### Dataset Format

Your dataset should be in JSONL format with the following structure:

```json
{"audio_filepath": "/path/to/audio.wav", "text": "transcription text", "duration": 3.2}
{"audio_filepath": "/path/to/audio2.wav", "text": "another transcription", "duration": 2.1}
```

### Key Training Parameters

- **Batch Configuration**: Adjust `dataset.train_loader.max_batch_duration` for memory management
- **GPU Settings**: Set `trainer.devices` and `trainer.accelerator`
- **Precision**: Use `trainer.precision=16` for mixed precision training
- **Checkpointing**: Models are saved automatically based on `callbacks.cb` settings

### Example Training Commands

```bash
# Multi-GPU training
python train.py trainer.devices=[0,1,2,3] trainer.strategy=ddp

# Resume from checkpoint
python train.py trainer.resume_from_checkpoint=/path/to/checkpoint.ckpt

# Train with custom dataset
python train.py dataset.train_ds.filepaths=[/data/my_train.jsonl] \
                dataset.val_ds.filepaths=[/data/my_val.jsonl]
