# 🎤 NeMo-Lite Speech Recognition Toolkit

> A streamlined, customizable speech recognition toolkit built on NeMo foundations with PyTorch Lightning power, featuring all state-of-the-art ASR algorithms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-purple.svg)](https://lightning.ai)
[![NeMo](https://img.shields.io/badge/NeMo-Compatible-green.svg)](https://github.com/NVIDIA/NeMo)
[![CUDA](https://img.shields.io/badge/CUDA-Optimized-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Triton](https://img.shields.io/badge/Triton-Supported-FF6B35.svg)](https://triton-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 What Makes This Special?

**NeMo-Lite** brings you the complete arsenal of NVIDIA NeMo's cutting-edge ASR algorithms in a simplified, customizable package. Built for researchers, developers, and ML engineers who demand both **performance** and **flexibility**.

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

