from typing import Tuple

import torch
from hydra.utils import instantiate
from jiwer import wer
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from ezspeech.modules.data.sampler import DynamicBatchSampler
from ezspeech.modules.data.utils.text import Tokenizer
from ezspeech.optims.scheduler import NoamAnnealing


class ASR_ctc_training(LightningModule):
    def __init__(self, config: DictConfig):
        super(ASR_ctc_training, self).__init__()

        self.save_hyperparameters()
        self.config = config
        self.preprocessor = instantiate(config.model.preprocessor)

        self.spec_augment = instantiate(config.model.spec_augment)

        self.encoder = instantiate(config.model.encoder)

        self.ctc_decoder = instantiate(config.model.decoder)

        self.ctc_loss = instantiate(config.model.loss.ctc_loss)

        # Initialize tokenizer for WER calculation
        self.tokenizer = Tokenizer(spe_file=config.dataset.spe_file)

        # Initialize WER accumulation for validation set
        self.val_predictions = []
        self.val_references = []

    def train_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.train_ds, _recursive_=False)
        dataset.set_tokenizer(self.tokenizer)
        sampler = SequentialSampler(dataset)
        # dataset=self.get_distributed_dataset(dataset)
        loader = self.hparams.config.dataset.train_loader

        dynamic_batcher = DynamicBatchSampler(
            sampler=sampler,
            max_batch_duration=loader.max_batch_duration,
            num_buckets=loader.num_bucket
        )
        train_dl = DataLoader(
            dataset=dataset,
            batch_sampler=dynamic_batcher,
            collate_fn=dataset.collate_asr_data,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory
            # shuffle=True,

        )
        return train_dl

    def val_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.val_ds, _recursive_=False)
        dataset.set_tokenizer(self.tokenizer)
        val_loader = self.hparams.config.dataset.val_loader
        sampler = DistributedSampler(dataset)
        val_dl = DataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=dataset.collate_asr_data,
            shuffle=False,
            **val_loader,
        )

        return val_dl

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        wavs, wav_lengths, targets, target_lengths = batch
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)
        loss = self._shared_step(features, feature_lengths, targets, target_lengths)
        self.log("loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        wavs, wav_lengths, targets, target_lengths = batch
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)

        loss = self._shared_step(features, feature_lengths, targets, target_lengths)

        # Calculate WER
        # Get CTC predictions for WER calculation
        enc_outs, enc_lens = self.encoder(features, feature_lengths)
        ctc_logits = self.ctc_decoder(enc_outs)

        # Decode predictions and targets to text
        predictions = self._ctc_decode_predictions(ctc_logits, enc_lens)
        references = self._targets_to_text(targets, target_lengths)

        # Accumulate predictions and references for validation set WER
        self.val_predictions.extend(predictions)
        self.val_references.extend(references)

        # Calculate WER for entire validation set so far
        validation_wer = wer(self.val_references, self.val_predictions)
        self.log("val_wer", validation_wer, sync_dist=True, prog_bar=True)

        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def on_validation_start(self):
        """
        Reset WER accumulation at the start of validation
        """
        self.val_predictions = []
        self.val_references = []

    def _shared_step(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

        enc_outs, enc_lens = self.encoder(inputs, input_lengths)
        ctc_logits = self.ctc_decoder(enc_outs)

        ctc_loss = self.ctc_loss(
            log_probs=ctc_logits,
            targets=targets,
            input_lengths=enc_lens,
            target_lengths=target_lengths,
        )

        return ctc_loss

    def _ctc_decode_predictions(self, logits, input_lengths):
        """
        Decode CTC predictions to text for WER calculation
        """
        # Get predictions using argmax
        predicted_ids = torch.argmax(logits, dim=-1)  # [B, T]

        predictions = []
        for i, pred_seq in enumerate(predicted_ids):
            # Get the actual length for this sequence
            seq_len = input_lengths[i].item()
            pred_seq = pred_seq[:seq_len]

            # Remove consecutive duplicates and blank tokens
            unique_seq = torch.unique_consecutive(pred_seq)
            # Remove blank token (usually the last token in vocab)
            filtered_seq = (
                unique_seq[unique_seq != len(self.tokenizer.vocab)]
                .cpu()
                .numpy()
                .tolist()
            )

            # Convert token IDs to text
            if len(filtered_seq) > 0:
                tokens = self.tokenizer.decode(filtered_seq)
                text = "".join(tokens).replace("_", " ").strip()
            else:
                text = ""

            predictions.append(text)

        return predictions

    def _targets_to_text(self, targets, target_lengths):
        """
        Convert target token sequences to text
        """
        references = []
        for i, target_seq in enumerate(targets):
            # Get the actual length for this sequence
            seq_len = target_lengths[i].item()
            target_seq = target_seq[:seq_len]

            # Convert token IDs to text
            tokens = self.tokenizer.decode(target_seq.cpu().numpy().tolist())
            text = "".join(tokens).replace("_", " ").strip()
            references.append(text)

        return references

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            **self.hparams.config.model.optimizer,
        )
        scheduler = NoamAnnealing(
            optimizer,
            **self.hparams.config.model.scheduler,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def export_checkpoint(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "preprocessor": self.preprocessor.state_dict(),
                "encoder": self.encoder.state_dict(),
                "ctc_decoder": self.ctc_decoder.state_dict(),
            },
            "hyper_parameters": self.hparams.config.model,
        }
        print("checkpoint")
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint is saved to "{filepath}" ...')
