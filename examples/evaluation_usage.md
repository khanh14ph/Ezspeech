# EzSpeech Evaluation Usage Guide

This guide shows how to use the improved evaluation system where all configuration is managed through YAML files instead of command line arguments.

## 🏗️ Configuration Structure

All evaluation parameters including model checkpoint and test files are now configured in YAML files under the `config/` directory.

### Available Configurations

1. **`config/eval.yaml`** - Template configuration with placeholder paths
2. **`config/test.yaml`** - Pre-configured test setup with actual paths

## 📋 Quick Start

### Option 1: Use Pre-configured Test Setup

```bash
# Run evaluation with pre-defined test configuration
python scripts/evaluate.py --config-name=test
```

### Option 2: Create Custom Configuration

```bash
# 1. Copy the template
cp config/eval.yaml config/my_eval.yaml

# 2. Edit your configuration
vim config/my_eval.yaml

# 3. Run evaluation
python scripts/evaluate.py --config-name=my_eval
```

### Option 3: Override from Command Line

```bash
# Override specific values while using base config
python scripts/evaluate.py --config-name=test \
  model_checkpoint=/path/to/different/model.ckpt \
  output_dir=custom_results \
  eval_batch_size=16
```

## ⚙️ Configuration File Format

### Required Fields

```yaml
# Model checkpoint path (required)
model_checkpoint: /path/to/your/model.ckpt

# Evaluation settings
eval_batch_size: 8
eval_num_workers: 4
output_dir: outputs/evaluation
```

### Dataset Evaluation

```yaml
eval_datasets:
  test_set:
    _target_: ezspeech.modules.data.dataset.SpeechRecognitionDataset
    filepaths:
      - /path/to/test.jsonl
    data_dir: /path/to/audio/files

  validation_set:
    _target_: ezspeech.modules.data.dataset.SpeechRecognitionDataset
    filepaths:
      - /path/to/val.jsonl
    data_dir: /path/to/audio/files
```

### Single File Evaluation

```yaml
eval_files:
  - audio_path: /path/to/sample1.wav
    reference_text: "hello world"
  - audio_path: /path/to/sample2.wav
    reference_text: "this is a test"
```

### Model Configuration

```yaml
# Dataset configuration (for tokenizer)
dataset:
  spe_file_grapheme: /path/to/grapheme.model
  spe_file_phoneme: /path/to/phoneme.model

# Model architecture (subset needed for evaluation)
model:
  d_model: 512
  vocab_size: 2048
  vocab_size_phoneme: 2048

  preprocessor:
    _target_: ezspeech.modules.data.utils.audio.AudioToMelSpectrogramPreprocessor
    sample_rate: 16000
    # ... other preprocessor settings
```

## 🔍 Example Configurations

### Minimal Configuration

```yaml
# config/minimal_eval.yaml
model_checkpoint: /path/to/model.ckpt
output_dir: results

eval_files:
  - audio_path: /path/to/test.wav
    reference_text: "test transcription"

dataset:
  spe_file_grapheme: /path/to/grapheme.model

model:
  d_model: 512
  vocab_size: 1024
```

### Dataset-only Evaluation

```yaml
# config/dataset_eval.yaml
model_checkpoint: /path/to/model.ckpt
output_dir: dataset_results

eval_datasets:
  librispeech_test:
    _target_: ezspeech.modules.data.dataset.SpeechRecognitionDataset
    filepaths:
      - /data/librispeech/test.jsonl
    data_dir: /data/librispeech/audio/

dataset:
  spe_file_grapheme: /path/to/grapheme.model

model:
  d_model: 512
  vocab_size: 2048
```

### Multi-dataset Evaluation

```yaml
# config/multi_eval.yaml
model_checkpoint: /path/to/model.ckpt
output_dir: multi_dataset_results

eval_datasets:
  librispeech_clean:
    _target_: ezspeech.modules.data.dataset.SpeechRecognitionDataset
    filepaths:
      - /data/librispeech/test_clean.jsonl
    data_dir: /data/librispeech/

  librispeech_noisy:
    _target_: ezspeech.modules.data.dataset.SpeechRecognitionDataset
    filepaths:
      - /data/librispeech/test_other.jsonl
    data_dir: /data/librispeech/

  custom_dataset:
    _target_: ezspeech.modules.data.dataset.SpeechRecognitionDataset
    filepaths:
      - /data/custom/test.jsonl
    data_dir: /data/custom/audio/
```

## 📊 Output

The evaluation script will:

1. **Validate Configuration**: Check if model checkpoint and test files exist
2. **Run Evaluation**: Process all configured datasets and files
3. **Generate Report**: Save detailed results to JSON file
4. **Log Progress**: Show real-time evaluation progress and metrics

### Output Structure

```
outputs/evaluation/
├── evaluation_results.json    # Complete results
├── detailed_metrics.json      # Per-sample results
└── evaluation_log.txt         # Execution log
```

## 🚨 Error Handling

The improved evaluation script includes:

- **Path Validation**: Checks if files exist before processing
- **Graceful Failures**: Continues evaluation even if some files fail
- **Detailed Logging**: Clear error messages and warnings
- **Partial Results**: Saves successful evaluations even if others fail

## 💡 Tips

1. **Start with Template**: Always copy from `config/eval.yaml` for new configurations
2. **Use Absolute Paths**: Avoid relative paths for better reliability
3. **Test Small First**: Start with a few files before running large evaluations
4. **Check Logs**: Monitor the console output for warnings and errors
5. **Validate Configs**: Use `python -c "import yaml; yaml.safe_load(open('config/your_config.yaml'))"` to check syntax

## 🔄 Migration from Old Usage

### Old Way (Command Line Arguments)
```bash
# OLD - Don't use anymore
python scripts/evaluate.py --config-name=eval \
  model_checkpoint=/path/to/model.ckpt \
  eval_datasets.test_set.filepaths=[/path/to/test.jsonl]
```

### New Way (Configuration File)
```bash
# NEW - Recommended approach
# 1. Update config/test.yaml with your paths
# 2. Run evaluation
python scripts/evaluate.py --config-name=test
```

This approach is much cleaner, more maintainable, and allows for complex evaluation setups that would be difficult to specify on the command line.