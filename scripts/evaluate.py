#!/usr/bin/env python3
"""
Comprehensive evaluation script for EzSpeech ASR models.
Supports multiple metrics and detailed analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torchaudio
from jiwer import cer, wer
from omegaconf import DictConfig
from tqdm import tqdm

from ezspeech.models.ctc_recognition import ASR_ctc_training
from ezspeech.utils.metrics import ASRMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASREvaluator:
    """Comprehensive ASR evaluation class."""

    def __init__(self, model_path: str, config: DictConfig):
        """Initialize evaluator with model and configuration."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        logger.info(f"Loading model from: {model_path}")
        self.model = ASR_ctc_training.load_from_checkpoint(model_path, config=config)
        self.model.eval()
        self.model.to(self.device)

        self.config = config
        self.metrics = ASRMetrics()

    def evaluate_dataset(self, dataset_config: DictConfig) -> Dict:
        """Evaluate model on a dataset."""
        from hydra.utils import instantiate
        from torch.utils.data import DataLoader

        # Create dataset
        dataset = instantiate(dataset_config, _recursive_=False)
        if hasattr(self.model, 'tokenizer_grapheme'):
            dataset.set_tokenizer(self.model.tokenizer_grapheme, self.model.tokenizer_phoneme)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("eval_batch_size", 8),
            num_workers=self.config.get("eval_num_workers", 4),
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        )

        predictions = []
        references = []

        logger.info(f"Evaluating on {len(dataset)} samples...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                            for k, v in batch.items()}

                # Get model predictions
                batch_predictions = self._get_predictions(batch)
                batch_references = self._get_references(batch)

                predictions.extend(batch_predictions)
                references.extend(batch_references)

        # Calculate metrics
        results = self._calculate_metrics(predictions, references)
        return results

    def _get_predictions(self, batch: Dict) -> List[str]:
        """Get model predictions for a batch."""
        # This will depend on your specific model implementation
        # For now, returning placeholder
        # You'll need to implement based on your model's forward pass
        return ["placeholder prediction"] * len(batch.get('audio', [1]))

    def _get_references(self, batch: Dict) -> List[str]:
        """Get reference transcriptions for a batch."""
        # Extract reference texts from batch
        if 'text' in batch:
            return batch['text']
        return ["placeholder reference"] * len(batch.get('audio', [1]))

    def _calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        results = {
            "total_samples": len(predictions),
            "word_error_rate": wer(references, predictions),
            "character_error_rate": cer(references, references),
        }

        # Additional metrics
        results.update(self.metrics.calculate_detailed_metrics(predictions, references))

        return results

    def evaluate_single_audio(self, audio_path: str, reference_text: Optional[str] = None) -> Dict:
        """Evaluate model on a single audio file."""
        logger.info(f"Evaluating single audio: {audio_path}")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Preprocess if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Move to device
        waveform = waveform.to(self.device)

        with torch.no_grad():
            # Get prediction (implement based on your model)
            prediction = self._predict_single(waveform)

        result = {
            "audio_path": audio_path,
            "prediction": prediction,
        }

        if reference_text:
            result["reference"] = reference_text
            result["wer"] = wer([reference_text], [prediction])
            result["cer"] = cer([reference_text], [prediction])

        return result

    def _predict_single(self, waveform: torch.Tensor) -> str:
        """Predict transcription for a single audio waveform."""
        # Implement based on your model's inference method
        # This is a placeholder
        return "placeholder prediction"


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(config: DictConfig) -> None:
    """Main evaluation function."""
    logger.info("Starting ASR evaluation...")

    # Initialize evaluator
    evaluator = ASREvaluator(
        model_path=config.model_checkpoint,
        config=config
    )

    results = {}

    # Evaluate on test datasets
    if hasattr(config, 'eval_datasets'):
        for dataset_name, dataset_config in config.eval_datasets.items():
            logger.info(f"Evaluating on dataset: {dataset_name}")
            dataset_results = evaluator.evaluate_dataset(dataset_config)
            results[dataset_name] = dataset_results

            logger.info(f"Results for {dataset_name}:")
            logger.info(f"  WER: {dataset_results['word_error_rate']:.4f}")
            logger.info(f"  CER: {dataset_results['character_error_rate']:.4f}")

    # Evaluate single files if specified
    if hasattr(config, 'eval_files'):
        single_file_results = []
        for file_config in config.eval_files:
            result = evaluator.evaluate_single_audio(
                audio_path=file_config.audio_path,
                reference_text=file_config.get('reference_text')
            )
            single_file_results.append(result)

        results['single_files'] = single_file_results

    # Save results
    output_dir = Path(config.get('output_dir', 'outputs/evaluation'))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to: {results_file}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()