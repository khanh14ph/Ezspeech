#!/usr/bin/env python3

"""
Simple single-GPU inference script for ASR CTC model.

Usage:
    python ezspeech/script/inference_simple.py \
        --checkpoint path/to/checkpoint.pt \
        --tokenizer path/to/tokenizer.model \
        --audio_files audio1.wav audio2.wav audio3.wav

    or with a directory:
    python ezspeech/script/inference_simple.py \
        --checkpoint path/to/checkpoint.pt \
        --tokenizer path/to/tokenizer.model \
        --audio_dir path/to/audio/dir \
        --output results.json
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch

from ezspeech.models.ctc import ASR_ctc_inference


def find_audio_files(directory: str, extensions: List[str] = ['.wav', '.flac', '.mp3']) -> List[str]:
    """Recursively find audio files in directory"""
    audio_files = []
    directory = Path(directory)

    for ext in extensions:
        audio_files.extend(directory.rglob(f'*{ext}'))

    return [str(f) for f in sorted(audio_files)]


def run_inference(
    checkpoint_path: str,
    tokenizer_path: str,
    audio_files: List[str],
    output_file: str = None,
    device: str = None
):
    """Run inference on audio files"""

    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print(f"Loading model from: {checkpoint_path}")

    # Load model
    model = ASR_ctc_inference(
        filepath=checkpoint_path,
        device=device,
        tokenizer_path=tokenizer_path
    )

    print(f"Model loaded successfully")
    print(f"\nProcessing {len(audio_files)} audio files...")

    # Run inference
    results = []
    for audio_file in audio_files:
        print(f"Transcribing: {audio_file}")
        try:
            transcription = model.transcribe([audio_file])[0]
            result = {
                'audio_path': audio_file,
                'transcription': transcription
            }
            results.append(result)
            print(f"  -> {transcription}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'audio_path': audio_file,
                'transcription': None,
                'error': str(e)
            })

    # Save results
    if output_file:
        output_data = {
            'checkpoint': checkpoint_path,
            'tokenizer': tokenizer_path,
            'device': device,
            'total_files': len(audio_files),
            'results': results
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Simple inference for ASR CTC model',
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
        '--audio_files',
        type=str,
        nargs='*',
        help='List of audio files to transcribe'
    )

    parser.add_argument(
        '--audio_dir',
        type=str,
        help='Directory containing audio files (will process all .wav, .flac, .mp3 files)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Path to output JSON file (optional)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    args = parser.parse_args()

    # Get audio files
    audio_files = []

    if args.audio_files:
        audio_files.extend(args.audio_files)

    if args.audio_dir:
        audio_files.extend(find_audio_files(args.audio_dir))

    if not audio_files:
        parser.error("No audio files specified. Use --audio_files or --audio_dir")

    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        audio_files=audio_files,
        output_file=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()
