# Changelog

All notable changes to the EzSpeech project will be documented in this file.

## [2.0.0] - 2024-09-18

### 🚀 Major Refactoring Release

This is a major refactoring of the EzSpeech codebase to improve code organization, readability, and ease of use.

### ✨ Added

- **New Scripts Directory**: Organized main scripts under `scripts/` for better structure
  - `scripts/train.py`: Enhanced training script with better error handling and logging
  - `scripts/evaluate.py`: Comprehensive evaluation script with detailed metrics
  - `scripts/serve_websocket.py`: WebSocket server for real-time ASR inference

- **WebSocket Real-time Inference**: Complete WebSocket-based ASR service
  - Real-time audio streaming support
  - File-based transcription
  - Multiple client connection handling
  - Configurable chunk processing

- **Enhanced Utilities**:
  - `ezspeech/utils/training.py`: Training utilities and helpers
  - `ezspeech/utils/metrics.py`: Comprehensive ASR evaluation metrics
  - Better error handling and logging throughout

- **Examples and Documentation**:
  - `examples/websocket_client.py`: Complete WebSocket client with multiple demo modes
  - `examples/README.md`: Detailed usage examples
  - Comprehensive README with usage guides

- **Configuration Management**:
  - `config/eval.yaml`: Evaluation configuration template
  - Better configuration validation and error reporting

- **Development Tools**:
  - Enhanced `requirements.txt` with version pinning and optional dependencies
  - Better project structure documentation

### 🔄 Changed

- **Training Script**: `train.py` now redirects to `scripts/train.py` with deprecation warning
- **Project Structure**: Better separation of concerns with dedicated directories
- **Error Handling**: Improved error messages and logging throughout
- **Documentation**: Complete rewrite of README with detailed examples

### 🛠️ Improved

- **Code Organization**: Clear separation between core library and application scripts
- **Logging**: Consistent logging across all modules
- **Metrics**: More comprehensive evaluation metrics including error analysis
- **Configuration**: Better validation and documentation of config options

### 📋 Technical Details

- **Backward Compatibility**: Existing `train.py` still works but shows deprecation warning
- **Dependencies**: Updated requirements with better version management
- **Testing**: Framework for unit tests (to be expanded)

### 🔧 Migration Guide

For existing users:

1. **Training**: Use `python scripts/train.py` instead of `python train.py`
2. **Evaluation**: Use new `scripts/evaluate.py` for comprehensive evaluation
3. **Real-time Inference**: Use `scripts/serve_websocket.py` for deployment
4. **Configuration**: Update config files to use new evaluation format if needed

### 🎯 Next Steps

- Unit tests for all new modules
- Performance benchmarks
- Model export utilities
- Advanced streaming features

## [1.0.0] - 2024-09-12

### Initial Release

- Basic CTC and Transducer ASR models
- Conformer encoder architecture
- PyTorch Lightning integration
- Hydra configuration management
- Basic training and inference capabilities