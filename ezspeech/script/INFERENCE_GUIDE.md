# Inference Scripts for ASR CTC Model

This directory contains two inference scripts for the ASR CTC model:

1. **`inference_distributed.py`** - Multi-GPU distributed inference using torchrun
2. **`inference_simple.py`** - Simple single-GPU inference

## Prerequisites

Make sure you have:
- A trained model checkpoint (`.pt` file)
- A tokenizer model file (`.model` file)
- Audio files to transcribe

## 1. Simple Inference (Single GPU)

### Basic Usage

```bash
python ezspeech/script/inference_simple.py \
    --checkpoint path/to/checkpoint.pt \
    --tokenizer path/to/tokenizer.model \
    --audio_files audio1.wav audio2.wav audio3.wav
```

### Process Entire Directory

```bash
python ezspeech/script/inference_simple.py \
    --checkpoint path/to/checkpoint.pt \
    --tokenizer path/to/tokenizer.model \
    --audio_dir path/to/audio/directory \
    --output results.json
```

### Force CPU Usage

```bash
python ezspeech/script/inference_simple.py \
    --checkpoint path/to/checkpoint.pt \
    --tokenizer path/to/tokenizer.model \
    --audio_files audio.wav \
    --device cpu
```

## 2. Distributed Inference (Multi-GPU with torchrun)

### Single GPU (equivalent to simple mode)

```bash
python ezspeech/script/inference_distributed.py \
    --checkpoint path/to/checkpoint.pt \
    --tokenizer path/to/tokenizer.model \
    --input_jsonl dataset/demo.jsonl \
    --output_file predictions.json
```

### Multi-GPU with torchrun (4 GPUs)

```bash
torchrun --nproc_per_node=4 ezspeech/script/inference_distributed.py \
    --checkpoint path/to/checkpoint.pt \
    --tokenizer path/to/tokenizer.model \
    --input_jsonl dataset/demo.jsonl \
    --output_file predictions.json \
    --batch_size 8 \
    --num_workers 4
```

### Multi-GPU with Custom Settings

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    ezspeech/script/inference_distributed.py \
    --checkpoint path/to/checkpoint.pt \
    --tokenizer path/to/tokenizer.model \
    --input_jsonl dataset/test.jsonl \
    --data_dir /path/to/audio/files \
    --output_file results.json \
    --batch_size 16
```

## Input Format

### For `inference_distributed.py`

The script expects a JSONL file where each line is a JSON object with:

```json
{"audio_filepath": "path/to/audio.wav", "text": "reference transcription", "duration": 3.5}
```

Fields:
- `audio_filepath` (required): Path to audio file
- `text` (optional): Reference transcription for calculating WER/CER
- `duration` (optional): Audio duration in seconds

### For `inference_simple.py`

Just provide audio file paths directly via command line arguments.

## Output Format

### `inference_distributed.py` Output

```json
{
  "metrics": {
    "wer": 0.1234,
    "cer": 0.0567,
    "total_samples": 100
  },
  "checkpoint": "path/to/checkpoint.pt",
  "tokenizer": "path/to/tokenizer.model",
  "results": [
    {
      "id": 0,
      "audio_path": "audio1.wav",
      "prediction": "predicted transcription",
      "reference": "reference transcription",
      "duration": 3.5,
      "rank": 0
    }
  ]
}
```

### `inference_simple.py` Output

```json
{
  "checkpoint": "path/to/checkpoint.pt",
  "tokenizer": "path/to/tokenizer.model",
  "device": "cuda",
  "total_files": 3,
  "results": [
    {
      "audio_path": "audio1.wav",
      "transcription": "predicted transcription"
    }
  ]
}
```

## Performance Tips

### For Distributed Inference:

1. **Batch Size**: Start with batch_size=8 and adjust based on GPU memory
   - Larger batch size = faster processing but more memory
   - Monitor GPU memory usage: `nvidia-smi`

2. **Number of Workers**: Set to 4-8 for optimal data loading
   - Too many workers can cause overhead
   - Too few can bottleneck GPU

3. **Number of GPUs**: Scale linearly
   - 4 GPUs will process ~4x faster than 1 GPU
   - Total effective batch size = batch_size × nproc_per_node

### For Simple Inference:

1. Process files in batches manually for large datasets
2. Use GPU when available (automatic by default)
3. Consider using distributed inference for large-scale processing

## Examples

### Example 1: Quick Test on Single File

```bash
python ezspeech/script/inference_simple.py \
    --checkpoint models/best_model.pt \
    --tokenizer models/tokenizer.model \
    --audio_files test_audio.wav
```

### Example 2: Process Demo Dataset (Single GPU)

```bash
python ezspeech/script/inference_distributed.py \
    --checkpoint models/best_model.pt \
    --tokenizer models/tokenizer.model \
    --input_jsonl dataset/demo.jsonl \
    --output_file results/demo_predictions.json
```

### Example 3: Large-Scale Processing (4 GPUs)

```bash
torchrun --nproc_per_node=4 ezspeech/script/inference_distributed.py \
    --checkpoint models/best_model.pt \
    --tokenizer models/tokenizer.model \
    --input_jsonl dataset/test_set.jsonl \
    --output_file results/test_predictions.json \
    --batch_size 16 \
    --num_workers 8
```

### Example 4: Process Directory with Simple Script

```bash
python ezspeech/script/inference_simple.py \
    --checkpoint models/best_model.pt \
    --tokenizer models/tokenizer.model \
    --audio_dir recordings/ \
    --output transcriptions.json
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
--batch_size 4  # or even 2
```

### Too Slow

1. Use distributed inference with multiple GPUs
2. Increase batch size if memory allows
3. Increase num_workers for faster data loading

### Import Errors

Make sure you're running from the project root:
```bash
cd /Users/khanh/dev/Ezspeech
python -m ezspeech.script.inference_simple --help
```

Or install the package:
```bash
pip install -e .
```

## Notes

- The distributed script automatically handles data distribution across GPUs
- Results are gathered and saved only by rank 0 (master process)
- WER/CER metrics are calculated automatically if reference text is provided
- Both scripts support various audio formats (.wav, .flac, .mp3)
