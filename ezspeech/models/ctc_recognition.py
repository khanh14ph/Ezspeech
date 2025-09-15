from typing import Tuple

import torch
from hydra.utils import instantiate
from jiwer import wer
from typing import List, Tuple
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from ezspeech.modules.data.sampler import DynamicBatchSampler
from ezspeech.modules.data.utils.text import Tokenizer
from ezspeech.optims.scheduler import NoamAnnealing
from ezspeech.utils.common import load_module

class ASR_ctc_training(LightningModule):
    def __init__(self, config: DictConfig):
        super(ASR_ctc_training, self).__init__()

        self.save_hyperparameters()
        self.config = config
        self.preprocessor = instantiate(config.model.preprocessor)

        self.spec_augment = instantiate(config.model.spec_augment)

        self.encoder = instantiate(config.model.encoder)

        self.ctc_decoder = instantiate(config.model.ctc_decoder)
        self.ctc_decoder_phoneme = instantiate(config.model.ctc_decoder_phoneme)
        self.back_projector_sc = instantiate(config.model.back_projector_sc)
        self.back_projector_sc_phoneme = instantiate(
            config.model.back_projector_sc_phoneme
        )

        self.ctc_loss = instantiate(config.model.loss.ctc_loss)

        # Initialize tokenizer for WER calculation
        self.tokenizer_grapheme = Tokenizer(spe_file=config.dataset.spe_file_grapheme)
        self.tokenizer_phoneme = Tokenizer(spe_file=config.dataset.spe_file_phoneme)
        # Initialize WER accumulation for validation set
        self.val_predictions = []
        self.val_references = []

    def train_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.train_ds, _recursive_=False)
        dataset.set_tokenizer(self.tokenizer_grapheme, self.tokenizer_phoneme)
        sampler = SequentialSampler(dataset)
        # dataset=self.get_distributed_dataset(dataset)
        loader = self.hparams.config.dataset.train_loader

        dynamic_batcher = DynamicBatchSampler(
            sampler=sampler,
            max_batch_duration=loader.max_batch_duration,
            num_buckets=loader.num_bucket,
        )
        train_dl = DataLoader(
            dataset=dataset,
            batch_sampler=dynamic_batcher,
            collate_fn=dataset.collate_asr_data,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            # shuffle=True,
        )
        return train_dl

    def val_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.val_ds, _recursive_=False)
        dataset.set_tokenizer(self.tokenizer_grapheme)
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
        (
            wavs,
            wav_lengths,
            targets_grapheme,
            target_lengths_grapheme,
            targets_phoneme,
            target_lengths_phoneme,
        ) = batch
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)
        enc_outs, enc_lens, logits_lst, logits_lst_phoneme = self.encoder(
            features,
            feature_lengths,
            self.ctc_decoder,
            self.back_projector_sc,
            self.ctc_decoder_phoneme,
            self.back_projector_sc_phoneme,
        )
        ctc_logits = self.ctc_decoder(enc_outs)

        loss = self.ctc_loss(
            log_probs=ctc_logits,
            targets=targets_grapheme,
            input_lengths=enc_lens,
            target_lengths=target_lengths_grapheme,
        )
        self.log("loss_last_layer", loss, sync_dist=True, prog_bar=False)
        if len(logits_lst) > 0:
            intermediate_loss_grapheme = self.self_condition_step(
                logits_lst, enc_lens, targets_grapheme, target_lengths_grapheme
            )

            if len(logits_lst_phoneme) > 0:
                intermediate_loss_phoneme = self.self_condition_step(
                    logits_lst_phoneme,
                    enc_lens,
                    targets_phoneme,
                    target_lengths_phoneme,
                )
                loss = (
                    loss / 2
                    + (intermediate_loss_grapheme + intermediate_loss_phoneme) / 2
                )
            else:
                loss = (loss + intermediate_loss_grapheme) / 2
        self.log("loss", loss, sync_dist=True, prog_bar=True)
        
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        wavs, wav_lengths, targets, target_lengths = batch
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)
        # Calculate WER
        # Get CTC predictions for WER calculation
        enc_outs, enc_lens, _,_ = self.encoder(
            features,
            feature_lengths,
            self.ctc_decoder,
            self.back_projector_sc,
            self.ctc_decoder_phoneme,
            self.back_projector_sc_phoneme,
        )
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
        return None

    def on_validation_start(self):
        """
        Reset WER accumulation at the start of validation
        """
        self.val_predictions = []
        self.val_references = []

    def self_condition_step(
        self, logits_lst, enc_lens, targets: torch.Tensor, target_lengths: torch.Tensor
    ):
        loss_lst = []
        for logit in logits_lst:
            loss = self.ctc_loss(
                log_probs=logit,
                targets=targets,
                input_lengths=enc_lens,
                target_lengths=target_lengths,
            )
            loss_lst.append(loss)
        return sum(loss_lst) / len(loss_lst)

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
                unique_seq[unique_seq != len(self.tokenizer_grapheme.vocab)]
                .cpu()
                .numpy()
                .tolist()
            )

            # Convert token IDs to text
            if len(filtered_seq) > 0:
                tokens = self.tokenizer_grapheme.decode(filtered_seq)
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
            tokens = self.tokenizer_grapheme.decode(target_seq.cpu().numpy().tolist())
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
                "ctc_decoder_phoneme": self.ctc_decoder_phoneme.state_dict(),
                "back_projector_sc": self.back_projector_sc.state_dict(),   
                "back_projector_sc_phoneme": self.back_projector_sc_phoneme.state_dict(),
            },
            "hyper_parameters": self.hparams.config.model,
        }
        print("checkpoint")
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint is saved to "{filepath}" ...')
class ASR_ctc_inference(object):
    def __init__(
        self, filepath: str, device: str, tokenizer_path: str = None, decoding_cfg=None
    ):
        self.blank = 0
        self.beam_size = 5

        self.device = device
        self.tokenizer = Tokenizer(vocab_file=tokenizer_path)
        self.vocab = self.tokenizer.vocab
        (
            self.preprocessor,
            self.encoder,
            self.ctc_decoder,
            self.ctc_decoder_phoneme,
            self.back_projector_sc,
            self.back_projector_sc_phoneme,
        ) = self._load_checkpoint(filepath, device)
    def _load_checkpoint(self, filepath: str, device: str):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        hparams = checkpoint["hyper_parameters"]
        weights = checkpoint["state_dict"]

        preprocessor = load_module(
            hparams["preprocessor"], weights["preprocessor"], device
        )
        encoder = load_module(hparams["encoder"], weights["encoder"], device)

        ctc_decoder = load_module(
            hparams["ctc_decoder"], weights["ctc_decoder"], device
        )
        ctc_decoder_phoneme = load_module(
            hparams["ctc_decoder_phoneme"], weights["ctc_decoder_phoneme"], device
        )
        back_projector_sc = load_module(
            hparams["back_projector_sc"], weights["back_projector_sc"], device
        )
        back_projector_sc_phoneme = load_module(
            hparams["back_projector_sc_phoneme"], weights["back_projector_sc_phoneme"], device
        )   

        return preprocessor, encoder, ctc_decoder, ctc_decoder_phoneme,back_projector_sc, back_projector_sc_phoneme
    
    @torch.inference_mode()
    def forward_encoder(
        self, speeches: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs, x_lens = self.collate_wav(speeches)
        xs, x_lens = self.preprocessor(xs.to(self.device), x_lens.to(self.device))
        enc_outs, enc_lens, _, _ = self.encoder(xs, x_lens, self.ctc_decoder, self.back_projector_sc, self.ctc_decoder_phoneme, self.back_projector_sc_phoneme)
        return enc_outs, enc_lens
    
    def transcribe(self, audio_lst):
        audios = [torchaudio.load(i) for i in audio_lst]
        speeches = [i[0] for i in audios]
        sample_rates = [i[1] for i in audios]
        enc, enc_length = self.forward_encoder(speeches)
        res = self.ctc_decode(enc, enc_length)
        return res

    @torch.inference_mode()
    def ctc_decode(self, enc_outs: List[torch.Tensor], enc_lens: List[torch.Tensor]):
        logits = self.ctc_decoder(enc_outs)
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids = [torch.unique_consecutive(i) for i in predicted_ids]
        print("predicted_ids:", predicted_ids)
        predicted_tokens = [self.idx_to_token(i) for i in predicted_ids]
        predicted_transcripts = ["".join(i) for i in predicted_tokens]
        predicted_transcripts = [
            i.replace("_", " ").strip() for i in predicted_transcripts
        ]
        return predicted_transcripts

    def idx_to_token(self, lst):
        lst = [j for j in lst if j != len(self.vocab)]
        return [self.vocab[i] for i in lst]

    def collate_wav(
        self, speeches: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        wavs = [b[0] for b in speeches]
        wav_lengths = [torch.tensor(len(f)) for f in wavs]
        max_audio_len = max(wav_lengths).item()
        new_audio_signal = []
        for sig, sig_len in zip(wavs, wav_lengths):
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            new_audio_signal.append(sig)
        new_audio_signal = torch.stack(new_audio_signal)
        audio_lengths = torch.stack(wav_lengths)
        return new_audio_signal, audio_lengths


if __name__ == "__main__":

    config = OmegaConf.load("/home4/khanhnd/Ezspeech/config/test/test.yaml")
    model = instantiate(config.model)
    import torchaudio

    audio1 = "/home4/khanhnd/vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R181.wav"
    # THE ENGLISH FORWARDED TO THE FRENCH BASKETS OF FLOWERS OF WHICH THEY HAD MADE A PLENTIFUL PROVISION TO GREET THE ARRIVAL OF THE YOUNG PRINCESS THE FRENCH IN RETURN INVITED THE ENGLISH TO A SUPPER WHICH WAS TO BE GIVEN THE NEXT DAY
    # audio = [audio1]
    # audio2="/home4/tuannd/vbee-asr/self-condition-asr/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/test-clean/6930/75918/6930-75918-0000.flac"
    # print(model.transcribe_streaming(audio1))
    tex1 = model.transcribe([audio1])
    print(tex1)
    # print(model.transcribe(audio)[0]=="hear nothing thing so expezcaris flow boes theatre sus days country tele can never refer one'ssel as i have tou had little money and")

