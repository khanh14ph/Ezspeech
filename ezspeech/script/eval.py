#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
import torchaudio
from jiwer import wer, cer
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm



def load_test_data(jsonl_path: str, data_dir: str = "") -> List[Dict[str, Any]]:
    """Load test data from JSONL file"""
    test_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # Construct full audio path
            audio_path = os.path.join(data_dir, data['audio_filepath']) if data_dir else data['audio_filepath']
            test_data.append({
                'audio_path': audio_path,
                'text': data['text'],
                'duration': data.get('duration', 0)
            })
    return test_data


def evaluate_model(config_path: str, test_jsonl: str, data_dir: str = "", output_file: str = None) -> Dict[str, float]:
    """Evaluate ASR model using CTC inference"""

    # Load config
    config = OmegaConf.load(config_path)

    # Initialize model
    model = instantiate(config.model)

    # Load test data
    test_data = load_test_data(test_jsonl, data_dir)

    print(f"Loaded {len(test_data)} test samples")

    # Evaluate
    predictions = []
    references = []
    results = []

    print("Running evaluation...")
    for sample in tqdm(test_data):
        try:
            # Transcribe audio
            transcription = model.transcribe([sample['audio_path']])[0]

            predictions.append(transcription)
            references.append(sample['text'])

            # Store individual result
            result = {
                'audio_path': sample['audio_path'],
                'reference': sample['text'],
                'prediction': transcription,
                'duration': sample['duration']
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing {sample['audio_path']}: {e}")
            continue

    # Calculate metrics
    if predictions and references:
        word_error_rate = wer(references, predictions)
        character_error_rate = cer(references, predictions)

        metrics = {
            'wer': word_error_rate,
            'cer': character_error_rate,
            'total_samples': len(predictions),
            'processed_samples': len(results)
        }

        print(f"\nEvaluation Results:")
        print(f"WER: {word_error_rate:.4f}")
        print(f"CER: {character_error_rate:.4f}")
        print(f"Total samples: {len(test_data)}")
        print(f"Processed samples: {len(results)}")

        # Save detailed results if output file specified
        if output_file:
            output_data = {
                'metrics': metrics,
                'config_path': config_path,
                'test_data_path': test_jsonl,
                'results': results
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"Detailed results saved to: {output_file}")

        return metrics
    else:
        print("No valid predictions generated!")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASR CTC model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config file')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test JSONL file')
    parser.add_argument('--data_dir', type=str, default="",
                       help='Base directory for audio files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for detailed results (JSON)')

    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate_model(
        config_path=args.config,
        test_jsonl=args.test_data,
        data_dir=args.data_dir,
        output_file=args.output
    )

    return metrics


if __name__ == "__main__":
    main()