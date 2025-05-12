# from typing import Tuple
# from omegaconf import DictConfig
# from hydra.utils import instantiate
# from pytorch_lightning import LightningModule

# import torch
# import torch.nn.functional as F
# from torch.optim import AdamW
# from torch.utils.data import DataLoader

# from ezspeech.modules.dataset.dataset import collate_asr_data
# from ezspeech.optims.scheduler import NoamAnnealing


# class SpeechRecognitionTask(LightningModule):
#     def __init__(self, dataset: DictConfig, model: DictConfig):
#         super(SpeechRecognitionTask, self).__init__()
#         self.save_hyperparameters()

#         self.encoder = instantiate(model.encoder)

#         self.decoder = instantiate(model.decoder)

#         self.predictor = instantiate(model.predictor)
#         self.joint = instantiate(model.joint)

#         self.criterion = instantiate(model.criterion)

#     def train_dataloader(self) -> DataLoader:
#         dataset = instantiate(self.hparams.dataset.train_ds, _recursive_=False)
#         loaders = self.hparams.dataset.loaders

#         train_dl = DataLoader(
#             dataset=dataset,
#             collate_fn=collate_asr_data,
#             shuffle=True,
#             **loaders,
#         )

#         return train_dl

#     def val_dataloader(self) -> DataLoader:
#         dataset = instantiate(self.hparams.dataset.val_ds, _recursive_=False)
#         loaders = self.hparams.dataset.loaders

#         val_dl = DataLoader(
#             dataset=dataset,
#             collate_fn=collate_asr_data,
#             shuffle=False,
#             **loaders,
#         )

#         return val_dl

#     def training_step(
#         self, batch: Tuple[torch.Tensor, ...], batch_idx: int
#     ) -> torch.Tensor:

#         inputs, input_lengths, targets, target_lengths= batch
#         # print("inputs", inputs.shape)
#         # print("____________")
#         loss,ctc_loss,rnnt_loss = self._shared_step(
#             inputs, input_lengths, targets, target_lengths
#         )
#         self.log("train_ctc_loss", ctc_loss, sync_dist=True,prog_bar=True)
#         self.log("train_rnnt_loss", rnnt_loss, sync_dist=True,prog_bar=True)
#         self.log("train_loss", loss, sync_dist=True,prog_bar=True)

#         return loss

#     def validation_step(
#         self, batch: Tuple[torch.Tensor, ...], batch_idx: int
#     ) -> torch.Tensor:

#         inputs, input_lengths, targets, target_lengths= batch

#         loss,ctc_loss,rnnt_loss= self._shared_step(
#             inputs, input_lengths, targets, target_lengths
#         )

#         self.log("val_ctc_loss", ctc_loss, sync_dist=True)
#         self.log("val_rnnt_loss", rnnt_loss, sync_dist=True)
#         self.log("val_loss", loss, sync_dist=True, prog_bar=True)

#         return loss

#     def _shared_step(
#         self,
#         inputs: torch.Tensor,
#         input_lengths: torch.Tensor,
#         targets: torch.Tensor,
#         target_lengths: torch.Tensor,
#     ) -> Tuple[torch.Tensor, ...]:

#         enc_outs, enc_lens = self.encoder(inputs, input_lengths)
#         ctc_logits = self.decoder(enc_outs)

#         ys = F.pad(targets, (1, 0))
#         pred_outs, __ = self.predictor(ys)
#         rnnt_logits = self.joint(enc_outs, pred_outs)

#         loss,ctc_loss,rnnt_loss = self.criterion(
#             ctc_logits, rnnt_logits,enc_lens, targets, target_lengths
#         )

#         return loss,ctc_loss,rnnt_loss

#     def configure_optimizers(self):
#         optimizer = AdamW(
#             self.parameters(),
#             **self.hparams.model.optimizer,
#         )
#         scheduler = NoamAnnealing(
#             optimizer,
#             **self.hparams.model.scheduler,
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "step",
#             },
#         }

#     def export(self, filepath: str):
#         checkpoint = {
#             "state_dict": {
#                 "encoder": self.encoder.state_dict(),
#                 "decoder": self.decoder.state_dict(),
#                 "predictor": self.predictor.state_dict(),
#                 "joint": self.joint.state_dict(),
#             },
#             "hyper_parameters": self.hparams.model,
#         }
#         print("checkpoint")
#         torch.save(checkpoint, filepath)
#         print(f'Model checkpoint is saved to "{filepath}" ...')


from typing import Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ezspeech.modules.dataset.dataset import collate_asr_data
from ezspeech.optims.scheduler import NoamAnnealing
from ezspeech.modules.metric.wer import WER


class SpeechRecognitionTask(LightningModule):
    def __init__(self, dataset: DictConfig, model: DictConfig):
        super(SpeechRecognitionTask, self).__init__()

        self.save_hyperparameters()

        self.preprocessor = instantiate(model.preprocessor)

        self.spec_augment = instantiate(model.spec_augment)

        self.encoder = instantiate(model.encoder)

        self.decoder = instantiate(model.decoder)

        self.predictor = instantiate(model.predictor)

        self.joint = instantiate(model.joint)

        self.rnnt_loss = instantiate(model.loss.rnnt_loss)

        self.joint.set_loss(self.rnnt_loss)

        self.ctc_loss = instantiate(model.loss.ctc_loss)

        self.vocab = open(dataset.vocab).read().splitlines()

    def train_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.dataset.train_ds, _recursive_=False)
        loaders = self.hparams.dataset.loaders

        train_dl = DataLoader(
            dataset=dataset,
            collate_fn=collate_asr_data,
            shuffle=True,
            **loaders,
        )

        return train_dl

    def val_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.dataset.val_ds, _recursive_=False)
        loaders = self.hparams.dataset.loaders

        val_dl = DataLoader(
            dataset=dataset,
            collate_fn=collate_asr_data,
            shuffle=False,
            **loaders,
        )

        return val_dl

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:

        wavs, wav_lengths, targets, target_lengths = batch
        features,feature_lengths=self.preprocessor(wavs, wav_lengths)
        #feature (B,T,D)
        features= self.spec_augment(features,feature_lengths)
        loss, ctc_loss, rnnt_loss = self._shared_step(
            features, feature_lengths, targets, target_lengths
        )
        self.log("train_ctc_loss", ctc_loss, sync_dist=True, prog_bar=True)
        self.log("train_rnnt_loss", rnnt_loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        wavs, wav_lengths, targets, target_lengths = batch
        features,feature_lengths=self.preprocessor(wavs, wav_lengths)
        

        loss, ctc_loss, rnnt_loss = self._shared_step(
            features, feature_lengths, targets, target_lengths
        )

        self.log("val_ctc_loss", ctc_loss, sync_dist=True)
        self.log("val_rnnt_loss", rnnt_loss, sync_dist=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def _shared_step(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

        enc_outs, enc_lens = self.encoder(inputs, input_lengths)
        ctc_logits = self.decoder(enc_outs)

        decoder_outputs, target_length, states = self.predictor(
            targets=targets, target_length=target_lengths
        )
        rnnt_loss = self.joint(
            encoder_outputs=enc_outs,
            decoder_outputs=decoder_outputs,
            encoder_lengths=enc_lens,
            transcripts=targets,
            transcript_lengths=target_lengths,
        )

        ctc_loss = self.ctc_loss(
            log_probs=ctc_logits,
            targets=targets,
            input_lengths=enc_lens,
            target_lengths=target_lengths,
        )
        loss = 0.7 * ctc_loss + 0.3 * rnnt_loss

        return loss, ctc_loss, rnnt_loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            **self.hparams.model.optimizer,
        )
        scheduler = NoamAnnealing(
            optimizer,
            **self.hparams.model.scheduler,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def export(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "joint": self.joint.state_dict(),
            },
            "hyper_parameters": self.hparams.model,
        }
        print("checkpoint")
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint is saved to "{filepath}" ...')
