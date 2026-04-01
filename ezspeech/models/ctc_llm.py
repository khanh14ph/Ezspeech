from typing import List, Tuple

import torch
import torch.nn as nn
import torchaudio
from hydra.utils import instantiate
from jiwer import wer
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from ezspeech.modules.data.dataset import SpeechRecognitionDataset
from ezspeech.modules.data.utils.text import Tokenizer
from ezspeech.optims.scheduler import NoamAnnealing
from ezspeech.utils.common import load_module


class LinearPoolConnector(nn.Module):
    """Projects speech encoder output to LLM input dimension with temporal pooling."""

    def __init__(self, input_dim: int, output_dim: int, pool_factor: int = 2):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        self.pool = nn.AvgPool1d(kernel_size=pool_factor, stride=pool_factor)
        self.linear2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_enc]
        x = self.linear1(x)        # [B, T, d_llm]
        x = x.transpose(1, 2)     # [B, d_llm, T]
        x = self.pool(x)           # [B, d_llm, T']
        x = x.transpose(1, 2)     # [B, T', d_llm]
        x = self.linear2(x)
        return x


class SpeechLLMDataset(SpeechRecognitionDataset):
    """Extends SpeechRecognitionDataset to return LLM-tokenized prompt/response pairs."""

    def set_llm_tokenizer(
        self,
        llm_tokenizer: AutoTokenizer,
        pre_prompt: str = "",
        post_prompt: str = "Transcribe:",
    ):
        self.llm_tokenizer = llm_tokenizer
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        speech, _ = super().__getitem__(idx)
        transcript = self.dataset[idx]["text"]

        pre_ids = self.llm_tokenizer.encode(
            self.pre_prompt, add_special_tokens=True, return_tensors="pt"
        )[0]
        post_ids = self.llm_tokenizer.encode(
            self.post_prompt, add_special_tokens=False, return_tensors="pt"
        )[0]
        output_ids = self.llm_tokenizer.encode(
            transcript, add_special_tokens=False, return_tensors="pt"
        )[0]
        eos = torch.tensor([self.llm_tokenizer.eos_token_id], dtype=torch.long)
        output_ids = torch.cat([output_ids, eos])

        return speech, pre_ids, post_ids, output_ids

    def collate_llm_data(self, batch: List) -> Tuple[torch.Tensor, ...]:
        wavs = [b[0][0] for b in batch]
        wav_lengths = [torch.tensor(len(w)) for w in wavs]
        max_wav_len = max(wav_lengths).item()

        padded_wavs = []
        for sig, sig_len in zip(wavs, wav_lengths):
            if sig_len < max_wav_len:
                sig = nn.functional.pad(sig, (0, max_wav_len - sig_len))
            padded_wavs.append(sig)
        padded_wavs = torch.stack(padded_wavs)
        wav_lengths = torch.stack(wav_lengths)

        pad_id = self.llm_tokenizer.pad_token_id or 0
        pre_ids = _pad_sequence([b[1] for b in batch], pad_id)
        post_ids = _pad_sequence([b[2] for b in batch], pad_id)
        output_ids = _pad_sequence([b[3] for b in batch], -100)

        return padded_wavs, wav_lengths, pre_ids, post_ids, output_ids


def _pad_sequence(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return padded


class ASR_ctc_llm_training(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Speech front-end (same architecture as CTC model)
        self.preprocessor = instantiate(config.model.preprocessor)
        self.spec_augment = instantiate(config.model.spec_augment)
        self.encoder = instantiate(config.model.encoder)

        # Connector: speech encoder dim -> LLM dim
        connector_cfg = config.model.connector
        self.connector = LinearPoolConnector(
            input_dim=connector_cfg.enc_dim,
            output_dim=connector_cfg.llm_dim,
            pool_factor=connector_cfg.get("pool_factor", 2),
        )

        # LLM backbone
        llm_cfg = config.model.llm
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_cfg.name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_cfg.name,
            trust_remote_code=True,
        )

        # Freeze encoder if not finetuning
        if not config.model.get("finetune_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Freeze LLM if not finetuning (useful when only training the connector)
        if not config.model.get("finetune_llm", True):
            for param in self.llm.parameters():
                param.requires_grad = False

        # Load pretrained weights if specified
        if config.model.get("model_pretrained") and config.model.model_pretrained.get("path"):
            self._load_pretrained(
                config.model.model_pretrained.path,
                config.model.model_pretrained.get("include", ["encoder"]),
            )

        self.val_predictions = []
        self.val_references = []

    def _load_pretrained(self, path: str, include: List[str]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        weights = checkpoint.get("state_dict", checkpoint)
        for key in include:
            if key in weights:
                getattr(self, key).load_state_dict(weights[key])
                print(f"Loaded pretrained weights for: {key}")

    def _embed_speech(
        self, wavs: torch.Tensor, wav_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, feat_lengths = self.preprocessor(wavs, wav_lengths)
        enc_out, enc_lens = self.encoder(features, feat_lengths)
        speech_embeds = self.connector(enc_out)
        return speech_embeds, enc_lens

    def _build_inputs(
        self,
        speech_embeds: torch.Tensor,
        pre_ids: torch.Tensor,
        post_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedder = self.llm.get_input_embeddings()

        pre_embeds = embedder(pre_ids.clamp(min=0))
        post_embeds = embedder(post_ids.clamp(min=0))
        out_embeds = embedder(output_ids.clamp(min=0))

        # [pre_prompt | speech | post_prompt | response]
        combined = torch.cat([pre_embeds, speech_embeds, post_embeds, out_embeds], dim=1)
        atts = torch.ones(combined.shape[:-1], dtype=torch.long, device=combined.device)

        input_len = pre_ids.shape[1] + speech_embeds.shape[1] + post_ids.shape[1]
        batch_size = combined.shape[0]
        labels = torch.cat(
            [
                torch.full(
                    (batch_size, input_len), -100, device=combined.device, dtype=torch.long
                ),
                output_ids,
            ],
            dim=1,
        )
        return combined, atts, labels

    def _make_dataset(self, ds_cfg) -> SpeechLLMDataset:
        dataset = SpeechLLMDataset(
            filepaths=ds_cfg.filepaths,
            data_dir=ds_cfg.data_dir,
        )
        dataset.set_tokenizer(Tokenizer(spe_file=self.hparams.config.dataset.spe_file))
        dataset.set_llm_tokenizer(
            self.llm_tokenizer,
            pre_prompt=self.hparams.config.dataset.get("pre_prompt", ""),
            post_prompt=self.hparams.config.dataset.get("post_prompt", "Transcribe:"),
        )
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataset = self._make_dataset(self.hparams.config.dataset.train_ds)
        loader_cfg = self.hparams.config.dataset.train_loader
        return DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_llm_data,
            shuffle=True,
            **loader_cfg,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._make_dataset(self.hparams.config.dataset.val_ds)
        val_loader_cfg = self.hparams.config.dataset.val_loader
        sampler = DistributedSampler(dataset)
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=dataset.collate_llm_data,
            shuffle=False,
            **val_loader_cfg,
        )

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        wavs, wav_lengths, pre_ids, post_ids, output_ids = batch
        speech_embeds, _ = self._embed_speech(wavs, wav_lengths)
        combined, atts, labels = self._build_inputs(speech_embeds, pre_ids, post_ids, output_ids)
        out = self.llm(inputs_embeds=combined, attention_mask=atts, labels=labels)
        self.log("train_loss", out.loss, sync_dist=True, prog_bar=True)
        return out.loss

    def validation_step(self, batch: Tuple, batch_idx: int):
        wavs, wav_lengths, pre_ids, post_ids, output_ids = batch
        speech_embeds, _ = self._embed_speech(wavs, wav_lengths)
        combined, atts, labels = self._build_inputs(speech_embeds, pre_ids, post_ids, output_ids)
        out = self.llm(inputs_embeds=combined, attention_mask=atts, labels=labels)
        self.log("val_loss", out.loss, sync_dist=True, prog_bar=True)

        input_len = pre_ids.shape[1] + speech_embeds.shape[1] + post_ids.shape[1]
        predicted_ids = torch.argmax(out.logits, dim=-1)
        for i in range(wavs.shape[0]):
            pred = self.llm_tokenizer.decode(
                predicted_ids[i, input_len:], skip_special_tokens=True
            )
            ref_ids = output_ids[i]
            ref_ids = ref_ids[ref_ids != -100]
            ref = self.llm_tokenizer.decode(ref_ids.cpu(), skip_special_tokens=True)
            self.val_predictions.append(pred)
            self.val_references.append(ref)

        if self.val_references:
            val_wer = wer(self.val_references, self.val_predictions)
            self.log("val_wer", val_wer, sync_dist=True, prog_bar=True)

    def on_validation_start(self):
        self.val_predictions = []
        self.val_references = []

    def configure_optimizers(self):
        optimizer_cfg = self.config.model.optimizer
        param_groups = [
            {
                "params": self.encoder.parameters(),
                "lr": optimizer_cfg.get("encoder_lr", 1e-5),
            },
            {"params": self.connector.parameters(), "lr": optimizer_cfg.lr},
            {"params": self.llm.parameters(), "lr": optimizer_cfg.lr},
        ]
        optimizer = AdamW(
            param_groups,
            betas=optimizer_cfg.betas,
            weight_decay=optimizer_cfg.weight_decay,
            eps=optimizer_cfg.eps,
        )
        scheduler = NoamAnnealing(optimizer, **self.config.model.scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def export_checkpoint(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "preprocessor": self.preprocessor.state_dict(),
                "encoder": self.encoder.state_dict(),
                "connector": self.connector.state_dict(),
                "llm": self.llm.state_dict(),
            },
            "hyper_parameters": self.hparams.config.model,
        }
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved to "{filepath}"')


class ASR_ctc_llm_inference:
    def __init__(
        self,
        filepath: str,
        device: str,
        pre_prompt: str = "",
        post_prompt: str = "Transcribe:",
        max_new_tokens: int = 200,
    ):
        self.device = device
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.max_new_tokens = max_new_tokens

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        hparams = checkpoint["hyper_parameters"]
        weights = checkpoint["state_dict"]

        self.preprocessor = load_module(hparams["preprocessor"], weights["preprocessor"], device)
        self.encoder = load_module(hparams["encoder"], weights["encoder"], device)

        self.connector = LinearPoolConnector(
            input_dim=hparams["connector"]["enc_dim"],
            output_dim=hparams["connector"]["llm_dim"],
            pool_factor=hparams["connector"].get("pool_factor", 2),
        )
        self.connector.load_state_dict(weights["connector"])
        self.connector.eval().to(device)

        llm_name = hparams["llm"]["name"]
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True)
        self.llm.load_state_dict(weights["llm"])
        self.llm.eval().to(device)

    @torch.inference_mode()
    def transcribe(self, audio_lst: List[str]) -> List[str]:
        audios = [torchaudio.load(p) for p in audio_lst]
        speeches = [a[0] for a in audios]

        wavs, wav_lengths = self._collate_wav(speeches)
        wavs, wav_lengths = wavs.to(self.device), wav_lengths.to(self.device)

        features, feat_lengths = self.preprocessor(wavs, wav_lengths)
        enc_out, _ = self.encoder(features, feat_lengths)
        speech_embeds = self.connector(enc_out)

        embedder = self.llm.get_input_embeddings()
        pre_ids = self.llm_tokenizer.encode(
            self.pre_prompt, add_special_tokens=True, return_tensors="pt"
        ).to(self.device)
        post_ids = self.llm_tokenizer.encode(
            self.post_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        results = []
        for i in range(speech_embeds.shape[0]):
            pre_embeds = embedder(pre_ids)
            post_embeds = embedder(post_ids)
            speech_embed_i = speech_embeds[i : i + 1]
            combined = torch.cat([pre_embeds, speech_embed_i, post_embeds], dim=1)

            attention_mask = torch.ones(combined.shape[:-1], dtype=torch.long, device=self.device)
            generated = self.llm.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )
            text = self.llm_tokenizer.decode(generated[0], skip_special_tokens=True)
            results.append(text)

        return results

    @torch.inference_mode()
    def _collate_wav(
        self, speeches: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wavs = [b[0] for b in speeches]
        wav_lengths = [torch.tensor(len(f)) for f in wavs]
        max_len = max(wav_lengths).item()
        padded = []
        for sig, sig_len in zip(wavs, wav_lengths):
            if sig_len < max_len:
                sig = nn.functional.pad(sig, (0, max_len - sig_len))
            padded.append(sig)
        return torch.stack(padded), torch.stack(wav_lengths)
