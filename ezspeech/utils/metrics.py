"""
Evaluation metrics and utilities for ASR models.
"""

import logging
import re
from typing import Dict, List, Tuple

import torch
from jiwer import cer, wer

logger = logging.getLogger(__name__)


class ASRMetrics:
    """Comprehensive ASR evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.all_predictions = []
        self.all_references = []

    def add_batch(self, predictions: List[str], references: List[str]):
        """Add a batch of predictions and references."""
        assert len(predictions) == len(references), "Predictions and references must have same length"
        self.all_predictions.extend(predictions)
        self.all_references.extend(references)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.all_predictions or not self.all_references:
            return {}

        return self.calculate_detailed_metrics(self.all_predictions, self.all_references)

    @staticmethod
    def calculate_detailed_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate detailed ASR metrics."""
        if not predictions or not references:
            return {}

        metrics = {}

        # Basic metrics
        try:
            metrics['word_error_rate'] = wer(references, predictions)
            metrics['character_error_rate'] = cer(references, predictions)
        except Exception as e:
            logger.warning(f"Error calculating WER/CER: {e}")
            metrics['word_error_rate'] = 1.0
            metrics['character_error_rate'] = 1.0

        # Word-level accuracy
        metrics['word_accuracy'] = 1.0 - metrics['word_error_rate']
        metrics['character_accuracy'] = 1.0 - metrics['character_error_rate']

        # Sentence-level accuracy
        correct_sentences = sum(1 for pred, ref in zip(predictions, references)
                              if pred.strip().lower() == ref.strip().lower())
        metrics['sentence_accuracy'] = correct_sentences / len(predictions)

        # Length statistics
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]

        metrics['avg_prediction_length'] = sum(pred_lengths) / len(pred_lengths)
        metrics['avg_reference_length'] = sum(ref_lengths) / len(ref_lengths)
        metrics['length_ratio'] = metrics['avg_prediction_length'] / metrics['avg_reference_length']

        # Detailed error analysis
        error_analysis = ASRMetrics._analyze_errors(predictions, references)
        metrics.update(error_analysis)

        return metrics

    @staticmethod
    def _analyze_errors(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Analyze different types of errors."""
        substitutions = 0
        insertions = 0
        deletions = 0
        total_words = 0

        for pred, ref in zip(predictions, references):
            pred_words = pred.strip().split()
            ref_words = ref.strip().split()

            total_words += len(ref_words)

            # Simple error counting (this is a simplified version)
            # For more accurate counting, you'd want to use edit distance alignment
            len_diff = len(pred_words) - len(ref_words)

            if len_diff > 0:
                insertions += len_diff
            elif len_diff < 0:
                deletions += abs(len_diff)

            # Count substitutions (simplified)
            min_len = min(len(pred_words), len(ref_words))
            for i in range(min_len):
                if pred_words[i].lower() != ref_words[i].lower():
                    substitutions += 1

        if total_words > 0:
            return {
                'substitution_rate': substitutions / total_words,
                'insertion_rate': insertions / total_words,
                'deletion_rate': deletions / total_words,
            }
        else:
            return {
                'substitution_rate': 0.0,
                'insertion_rate': 0.0,
                'deletion_rate': 0.0,
            }

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for evaluation."""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    @staticmethod
    def calculate_confidence_metrics(predictions: List[str], confidences: List[float],
                                   references: List[str]) -> Dict[str, float]:
        """Calculate confidence-related metrics."""
        if len(predictions) != len(confidences) or len(predictions) != len(references):
            return {}

        # Calculate accuracy at different confidence thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        metrics = {}

        for threshold in thresholds:
            high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= threshold]

            if high_conf_indices:
                high_conf_preds = [predictions[i] for i in high_conf_indices]
                high_conf_refs = [references[i] for i in high_conf_indices]

                acc = sum(1 for pred, ref in zip(high_conf_preds, high_conf_refs)
                         if pred.strip().lower() == ref.strip().lower()) / len(high_conf_preds)

                metrics[f'accuracy_at_conf_{threshold}'] = acc
                metrics[f'coverage_at_conf_{threshold}'] = len(high_conf_indices) / len(predictions)

        # Average confidence
        metrics['average_confidence'] = sum(confidences) / len(confidences)

        return metrics


class TokenLevelMetrics:
    """Token-level evaluation metrics."""

    @staticmethod
    def calculate_token_accuracy(predicted_tokens: torch.Tensor,
                               target_tokens: torch.Tensor,
                               ignore_index: int = -100) -> float:
        """Calculate token-level accuracy."""
        mask = target_tokens != ignore_index
        correct = (predicted_tokens == target_tokens) & mask
        return correct.sum().float() / mask.sum().float()

    @staticmethod
    def calculate_perplexity(log_probs: torch.Tensor,
                           targets: torch.Tensor,
                           ignore_index: int = -100) -> float:
        """Calculate perplexity from log probabilities."""
        mask = targets != ignore_index
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll = nll * mask.float()

        total_nll = nll.sum()
        total_tokens = mask.sum()

        if total_tokens > 0:
            avg_nll = total_nll / total_tokens
            return torch.exp(avg_nll).item()
        else:
            return float('inf')


def print_metrics_table(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Print metrics in a formatted table."""
    print(f"\n{title}")
    print("=" * len(title))

    # Group metrics by type
    error_metrics = {k: v for k, v in metrics.items() if 'error' in k or 'rate' in k}
    accuracy_metrics = {k: v for k, v in metrics.items() if 'accuracy' in k}
    other_metrics = {k: v for k, v in metrics.items()
                    if k not in error_metrics and k not in accuracy_metrics}

    def print_metric_group(group_metrics: Dict[str, float], group_name: str):
        if group_metrics:
            print(f"\n{group_name}:")
            for metric, value in group_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric:25s}: {value:.4f}")
                else:
                    print(f"  {metric:25s}: {value}")

    print_metric_group(accuracy_metrics, "Accuracy Metrics")
    print_metric_group(error_metrics, "Error Metrics")
    print_metric_group(other_metrics, "Other Metrics")

    print()  # Empty line at the end