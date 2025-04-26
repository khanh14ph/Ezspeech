"""Decoders and output normalization for CTC.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Sung-Lin Yeh 2020
 * Adel Moumen 2023, 2024
"""

import dataclasses
import heapq
import math
import warnings
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclasses.dataclass
class CTCHypothesis:
    """This class is a data handler over the generated hypotheses.

    This class is the default output of the CTC beam searchers.

    It can be re-used for other decoders if using
    the beam searchers in an online fashion.

    Arguments
    ---------
    text : str
        The text of the hypothesis.
    last_lm_state : None
        The last LM state of the hypothesis.
    score : float
        The score of the hypothesis.
    lm_score : float
        The LM score of the hypothesis.
    text_frames : List[Tuple[str, Tuple[int, int]]], optional
        The list of the text and the corresponding frames.
    """

    text: str
    last_lm_state: None
    score: float
    lm_score: float
    text_frames: list = None


class TorchAudioCTCPrefixBeamSearcher:
    """TorchAudio CTC Prefix Beam Search Decoder.

    This class is a wrapper around the CTC decoder from TorchAudio. It provides a simple interface
    where you can either use the CPU or CUDA CTC decoder.

    The CPU decoder is slower but uses less memory. The CUDA decoder is faster but uses more memory.
    The CUDA decoder is also only available in the nightly version of torchaudio.

    A lot of features are missing in the CUDA decoder, such as the ability to use a language model,
    constraint search, and more. If you want to use those features, you have to use the CPU decoder.

    For more information about the CPU decoder, please refer to the documentation of TorchAudio:
    https://pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html

    For more information about the CUDA decoder, please refer to the documentation of TorchAudio:
    https://pytorch.org/audio/main/generated/torchaudio.models.decoder.cuda_ctc_decoder.html#torchaudio.models.decoder.cuda_ctc_decoder

    If you want to use the language model, or the lexicon search, please make sure that your
    tokenizer/acoustic model uses the same tokens as the language model/lexicon. Otherwise, the decoding will fail.

    The implementation is compatible with SentencePiece Tokens.

    Note: When using CUDA CTC decoder, the blank_index has to be 0. Furthermore, using CUDA CTC decoder
    requires the nightly version of torchaudio and a lot of VRAM memory (if you want to use a lot of beams).
    Overall, we do recommend to use the CTCBeamSearcher or CTCPrefixBeamSearcher in SpeechBrain if you wants to use
    n-gram + beam search decoding. If you wants to have constraint search, please use the CPU version of torchaudio,
    and if you want to speedup as much as possible the decoding, please use the CUDA version.

    Arguments
    ---------
    tokens : list or str
        The list of tokens or the path to the tokens file.
        If this is a path, then the file should contain one token per line.
    lexicon : str, default: None
        Lexicon file containing the possible words and corresponding spellings. Each line consists of a word and its space separated spelling.
        If None, uses lexicon-free decoding. (default: None)
    lm : str, optional
        A path containing KenLM language model or None if not using a language model. (default: None)
    lm_dict : str, optional
        File consisting of the dictionary used for the LM, with a word per line sorted by LM index.
        If decoding with a lexicon, entries in lm_dict must also occur in the lexicon file.
        If None, dictionary for LM is constructed using the lexicon file. (default: None)
    topk : int, optional
        Number of top CTCHypothesis to return. (default: 1)
    beam_size : int, optional
        Numbers of hypotheses to hold after each decode step. (default: 50)
    beam_size_token : int, optional
        Max number of tokens to consider at each decode step. If None, it is set to the total number of tokens. (default: None)
    beam_threshold : float, optional
        Threshold for pruning hypothesis. (default: 50)
    lm_weight : float, optional
        Weight of language model. (default: 2)
    word_score : float, optional
        Word insertion score. (default: 0)
    unk_score : float, optional
        Unknown word insertion score. (default: float("-inf"))
    sil_score : float, optional
        Silence insertion score. (default: 0)
    log_add : bool, optional
        Whether to use use logadd when merging hypotheses. (default: False)
    blank_index : int or str, optional
        Index of the blank token. If tokens is a file path, then this should be an str. Otherwise, this should be a int. (default: 0)
    sil_index : int or str, optional
        Index of the silence token. If tokens is a file path, then this should be an str. Otherwise, this should be a int. (default: 0)
    unk_word : str, optional
        Unknown word token. (default: "<unk>")
    using_cpu_decoder : bool, optional
        Whether to use the CPU searcher. If False, then the CUDA decoder is used. (default: True)
    blank_skip_threshold : float, optional
        Skip frames if log_prob(blank) > log(blank_skip_threshold), to speed up decoding (default: 1.0).
        Note: This is only used when using the CUDA decoder, and it might worsen the WER/CER results. Use it at your own risk.

    Example
    -------
    >>> import torch
    >>> from speechbrain.decoders import TorchAudioCTCPrefixBeamSearcher
    >>> probs = torch.tensor([[[0.2, 0.0, 0.8],
    ...                   [0.4, 0.0, 0.6]]])
    >>> log_probs = torch.log(probs)
    >>> lens = torch.tensor([1.0])
    >>> blank_index = 2
    >>> vocab_list = ['a', 'b', '-']
    >>> searcher = TorchAudioCTCPrefixBeamSearcher(tokens=vocab_list, blank_index=blank_index, sil_index=blank_index) # doctest: +SKIP
    >>> hyps = searcher(probs, lens) # doctest: +SKIP
    """

    def __init__(
        self,
        vocab: Union[list, str],
        lexicon: Optional[str] = None,
        lm: Optional[str] = None,
        lm_dict: Optional[str] = None,
        topk: int = 1,
        beam_size: int = 50,
        beam_size_token: Optional[int] = None,
        beam_threshold: float = 50,
        lm_weight: float = 2,
        word_score: float = 0,
        unk_score: float = float("-inf"),
        sil_score: float = 0,
        log_add: bool = False,
        blank_index: Union[str, int] = 0,
        sil_index: Union[str, int] = 0,
        unk_word: str = "<unk>",
        using_cpu_decoder: bool = True,
        blank_skip_threshold: float = 1.0,
    ):
        self.lexicon = lexicon
        self.tokens = [i.split("\t")[0] for i in open(vocab).read().splitlines()]
        self.lm = lm
        self.lm_dict = lm_dict
        self.topk = topk
        self.beam_size = beam_size
        self.beam_size_token = beam_size_token
        self.beam_threshold = beam_threshold
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.unk_score = unk_score
        self.sil_score = sil_score
        self.log_add = log_add
        self.blank_index = blank_index
        self.sil_index = sil_index
        self.unk_word = unk_word
        self.using_cpu_decoder = using_cpu_decoder
        self.blank_skip_threshold = blank_skip_threshold

        if self.using_cpu_decoder:
            try:
                from torchaudio.models.decoder import ctc_decoder
            except ImportError:
                raise ImportError(
                    "ctc_decoder not found. Please install torchaudio and flashlight to use this decoder."
                )

            # if this is a path, then torchaudio expect to be an index
            # while if its a list then it expects to be a token
            if isinstance(self.tokens, str):
                blank_token = self.blank_index
                sil_token = self.sil_index
            else:
                blank_token = self.tokens[self.blank_index]
                sil_token = self.tokens[self.sil_index]
            self._ctc_decoder = ctc_decoder(
                lexicon=self.lexicon,
                tokens=self.tokens,
                lm=self.lm,
                lm_dict=self.lm_dict,
                nbest=self.topk,
                beam_size=self.beam_size,
                beam_size_token=self.beam_size_token,
                beam_threshold=self.beam_threshold,
                lm_weight=self.lm_weight,
                word_score=self.word_score,
                unk_score=self.unk_score,
                sil_score=self.sil_score,
                log_add=self.log_add,
                blank_token=blank_token,
                sil_token=sil_token,
                unk_word=self.unk_word,
            )
        else:
            try:
                from torchaudio.models.decoder import cuda_ctc_decoder
            except ImportError:
                raise ImportError(
                    "cuda_ctc_decoder not found. Please install the latest version of torchaudio to use this decoder."
                )
            assert (
                self.blank_index == 0
            ), "Index of blank token has to be 0 when using CUDA CTC decoder."

            self._ctc_decoder = cuda_ctc_decoder(
                tokens=self.tokens,
                nbest=self.topk,
                beam_size=self.beam_size,
                blank_skip_threshold=self.blank_skip_threshold,
            )

    def decode_beams(
        self, log_probs: torch.Tensor, wav_len: Union[torch.Tensor, None] = None
    ) -> List[List[CTCHypothesis]]:
        """Decode log_probs using TorchAudio CTC decoder.

        If `using_cpu_decoder=True` then log_probs and wav_len are moved to CPU before decoding.
        When using CUDA CTC decoder, the timestep information is not available. Therefore, the timesteps
        in the returned hypotheses are set to None.

        Make sure that the input are in the log domain. The decoder will fail to decode
        logits or probabilities. The input should be the log probabilities of the CTC output.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the input audio.
            Shape: (batch_size, seq_length, vocab_size)
        wav_len : torch.Tensor, default: None
            The speechbrain-style relative length. Shape: (batch_size,)
            If None, then the length of each audio is assumed to be seq_length.

        Returns
        -------
        list of list of CTCHypothesis
            The decoded hypotheses. The outer list is over the batch dimension, and the inner list is over the topk dimension.
        """
        if wav_len is not None:
            wav_len = log_probs.size(1) * wav_len
        else:
            wav_len = torch.tensor(
                [log_probs.size(1)] * log_probs.size(0),
                device=log_probs.device,
                dtype=torch.int32,
            )

        if wav_len.dtype != torch.int32:
            wav_len = wav_len.to(torch.int32)

        if log_probs.dtype != torch.float32:
            raise ValueError("log_probs must be float32.")

        # When using CPU decoder, we need to move the log_probs and wav_len to CPU
        if self.using_cpu_decoder and log_probs.is_cuda:
            log_probs = log_probs.cpu()

        if self.using_cpu_decoder and wav_len.is_cuda:
            wav_len = wav_len.cpu()

        if not log_probs.is_contiguous():
            raise RuntimeError("log_probs must be contiguous.")

        results = self._ctc_decoder(log_probs, wav_len)
        print(results)
        tokens_preds = []
        words_preds = []
        scores_preds = []
        timesteps_preds = []

        # over batch dim
        for i in range(len(results)):
            if self.using_cpu_decoder:
                preds = [results[i][j].tokens.tolist() for j in range(len(results[i]))]
                preds = [[self.tokens[token] for token in tokens] for tokens in preds]
                tokens_preds.append(preds)

                timesteps = [
                    results[i][j].timesteps.tolist() for j in range(len(results[i]))
                ]
                timesteps_preds.append(timesteps)

            else:
                # no timesteps is available for CUDA CTC decoder
                timesteps = [None for _ in range(len(results[i]))]
                timesteps_preds.append(timesteps)

                preds = [results[i][j].tokens for j in range(len(results[i]))]
                preds = [[self.tokens[token] for token in tokens] for tokens in preds]
                tokens_preds.append(preds)

            words = [results[i][j].words for j in range(len(results[i]))]
            words_preds.append(words)

            scores = [results[i][j].score for j in range(len(results[i]))]
            scores_preds.append(scores)

        hyps = []
        for (
            batch_index,
            (batch_text, batch_score, batch_timesteps),
        ) in enumerate(zip(tokens_preds, scores_preds, timesteps_preds)):
            hyps.append([])
            for text, score, timestep in zip(batch_text, batch_score, batch_timesteps):
                hyps[batch_index].append(
                    CTCHypothesis(
                        text="".join(text),
                        last_lm_state=None,
                        score=score,
                        lm_score=score,
                        text_frames=timestep,
                    )
                )
        return hyps

    def __call__(
        self, log_probs: torch.Tensor, wav_len: Union[torch.Tensor, None] = None
    ) -> List[List[CTCHypothesis]]:
        """Decode log_probs using TorchAudio CTC decoder.

        If `using_cpu_decoder=True` then log_probs and wav_len are moved to CPU before decoding.
        When using CUDA CTC decoder, the timestep information is not available. Therefore, the timesteps
        in the returned hypotheses are set to None.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the input audio.
            Shape: (batch_size, seq_length, vocab_size)
        wav_len : torch.Tensor, default: None
            The speechbrain-style relative length. Shape: (batch_size,)
            If None, then the length of each audio is assumed to be seq_length.

        Returns
        -------
        list of list of CTCHypothesis
            The decoded hypotheses. The outer list is over the batch dimension, and the inner list is over the topk dimension.
        """
        return self.decode_beams(log_probs, wav_len)
