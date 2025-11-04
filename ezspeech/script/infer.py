#!/usr/bin/env python3

"""
Simple single-GPU inference script for ASR CTC model.

Usage:
    python ezspeech/script/inference_distributed.py \
        --checkpoint path/to/checkpoint.pt \
        --tokenizer path/to/tokenizer.model \
        --input_jsonl dataset/demo.jsonl \
        --output_file predictions.json \
        --batch_size 8
        --lexicon path/to/lexicon.txt
        --lm path/to/lm.arpa
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from ezspeech.models.ctc import ASR_ctc_inference


def load_samples(jsonl_path: str, data_dir: str = "") -> List[Dict[str, Any]]:
    """Load samples from JSONL file"""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            sample = json.loads(line.strip())
            # Construct full audio path
            audio_path = os.path.join(data_dir, sample['audio_filepath']) if data_dir else sample['audio_filepath']
            samples.append({
                'id': idx,
                'audio_path': audio_path,
                'text': sample.get('text', ''),
                'duration': sample.get('duration', 0)
            })
    return samples


def create_batches(samples: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Create batches from samples"""
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])
    return batches


def run_inference(
    checkpoint_path: str,
    tokenizer_path: str,
    input_jsonl: str,
    output_file: str,
    lexicon_path: str,
    lm_path: str,
    batch_size: int = 4,
):
    """Main inference function for single GPU"""

    # Set device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print(f"Running inference on device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_file}")
    print(f"Batch size: {batch_size}")

    # Load model
    print("\nLoading model...")
    model = ASR_ctc_inference(
        filepath=checkpoint_path,
        device=device,
        tokenizer_path=tokenizer_path,
        lexicon_path=lexicon_path,
        lm_path=lm_path
    )
    print("Model loaded successfully")

    # Load samples from JSONL
    print(f"\nLoading samples from {input_jsonl}...")
    samples = load_samples(input_jsonl, data_dir)
    print(f"Loaded {len(samples)} samples")

    # Create batches
    batches = create_batches(samples, batch_size)
    print(f"Created {len(batches)} batches")

    # Run inference
    all_results = []
    print("\nRunning inference...")

    for batch in tqdm(batches, desc="Processing batches"):
        try:
            # Extract audio paths
            audio_paths = [sample['audio_path'] for sample in batch]

            # Run inference
            transcriptions = model.transcribe_lm(audio_paths)

            # Store results
            for sample, transcription in zip(batch, transcriptions):
                result = {
                    'id': sample['id'],
                    'audio_path': sample['audio_path'],
                    'prediction': transcription,
                    'reference': sample['text'],
                    'duration': sample['duration']
                }
                all_results.append(result)

        except Exception as e:
            print(f"\nError processing batch: {e}")
            # Add failed results with empty predictions
            for sample in batch:
                result = {
                    'id': sample['id'],
                    'audio_path': sample['audio_path'],
                    'prediction': '',
                    'reference': sample['text'],
                    'duration': sample['duration'],
                    'error': str(e)
                }
                all_results.append(result)
            continue

    # Sort by ID to maintain original order
    all_results = sorted(all_results, key=lambda x: x['id'])

    # Calculate metrics if references are available
    metrics = {'total_samples': len(all_results)}

    if all_results and all_results[0]['reference']:
        try:
            from jiwer import wer, cer

            predictions = [r['prediction'] for r in all_results]
            references = [r['reference'] for r in all_results]

            word_error_rate = wer(references, predictions)
            character_error_rate = cer(references, predictions)

            print(f"\nMetrics:")
            print(f"WER: {word_error_rate:.4f}")
            print(f"CER: {character_error_rate:.4f}")

            metrics.update({
                'wer': word_error_rate,
                'cer': character_error_rate
            })
        except ImportError:
            print("\nWarning: jiwer not installed. Skipping WER/CER calculation.")

    # Save to file
    output_data = {
        'metrics': metrics,
        'checkpoint': checkpoint_path,
        'tokenizer': tokenizer_path,
        'results': all_results
    }

    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print(f"Total samples processed: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(
        description='Simple single-GPU inference for ASR CTC model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pt)'
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='Path to tokenizer model file'
    )

    parser.add_argument(
        '--input_jsonl',
        type=str,
        required=True,
        help='Path to input JSONL file with audio paths'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default='predictions.json',
        help='Path to output JSON file (default: predictions.json)'
    )
    parser.add_argument(
        '--lexicon_path',
        type=str,
        required=True,
        help='Path to lexicon file with audio paths'
    )
    parser.add_argument(
        '--lm_path',
        type=str,
        required=True,
        help='Path to lm paths'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )

    args = parser.parse_args()

    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        input_jsonl=args.input_jsonl,
        output_file=args.output_file,
        batch_size=args.batch_size,
        lexicon_path=args.lexicon_path,
        lm_path=args.lm_path
    )


if __name__ == "__main__":
    main()
