import copy
import re
import unicodedata
from abc import abstractmethod
from dataclasses import dataclass, field, is_dataclass
from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from ezspeech.modules.decoder.rnnt.rnnt_decoding import (
    rnnt_beam_decoding,
    rnnt_greedy_decoding,
    tdt_beam_decoding,
)
from ezspeech.modules.decoder.rnnt.rnnt_decoding.rnnt_batched_beam_utils import (
    BlankLMScoreMode,
    PruningMode,
)
from ezspeech.modules.decoder.rnnt.rnnt_utils import Hypothesis, NBestHypotheses


class RNNTDecoding:

    def __init__(
        self,
        decoding_cfg,
        decoder,
        joint,
        vocabulary,
    ):
        super(RNNTDecoding, self).__init__()

        # we need to ensure blank is the last token in the vocab for the case of RNNT and Multi-blank RNNT.
        blank_id = len(vocabulary)

        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])
        self.vocab_size = len(self.labels_map)
        # Convert dataclass to config object

        self.cfg = decoding_cfg
        self.blank_id = blank_id
        self.num_extra_outputs = joint.num_extra_outputs
        self.durations = self.cfg.get("durations", None)
        self.compute_hypothesis_token_set = self.cfg.get(
            "compute_hypothesis_token_set", False
        )

        self.tdt_include_token_duration = self.cfg.get(
            "tdt_include_token_duration", False
        )

        self._is_tdt = (
            self.durations is not None and self.durations != []
        )  # this means it's a TDT model.
        if self._is_tdt:
            if self.durations is None:
                raise ValueError("duration can't not be None")
            if self.cfg.strategy not in [
                "greedy",
                "greedy_batch",
                "beam",
                "maes",
                "malsd_batch",
            ]:
                raise ValueError(
                    "currently only greedy, greedy_batch, beam and maes inference is supported for TDT models"
                )

        possible_strategies = [
            "greedy",
            "greedy_batch",
            "beam",
            "alsd",
            "maes",
            "malsd_batch",
            "maes_batch",
        ]
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}")

        if self.cfg.strategy == "greedy":

            self.decoding = rnnt_greedy_decoding.GreedyTDTInfer(
                decoder_model=decoder,
                joint_model=joint,
                blank_index=self.blank_id,
                durations=self.durations,
                max_symbols_per_step=(
                    self.cfg.greedy.get("max_symbols", None)
                    or self.cfg.greedy.get("max_symbols_per_step", None)
                ),
                include_duration=self.tdt_include_token_duration,
            )

        elif self.cfg.strategy == "greedy_batch":
            self.decoding = rnnt_greedy_decoding.GreedyBatchedTDTInfer(
                decoder_model=decoder,
                joint_model=joint,
                blank_index=self.blank_id,
                durations=self.durations,
                max_symbols_per_step=(
                    self.cfg.greedy.get("max_symbols", None)
                    or self.cfg.greedy.get("max_symbols_per_step", None)
                ),
                include_duration=self.tdt_include_token_duration,
                use_cuda_graph_decoder=self.cfg.greedy.get(
                    "use_cuda_graph_decoder", True
                ),
                ngram_lm_model=self.cfg.greedy.get("ngram_lm_model", None),
                ngram_lm_alpha=self.cfg.greedy.get("ngram_lm_alpha", 0),
            )
        elif self.cfg.strategy == "beam":
            self.decoding = tdt_beam_decoding.BeamTDTInfer(
                decoder_model=decoder,
                joint_model=joint,
                durations=self.durations,
                beam_size=self.cfg.beam.beam_size,
                return_best_hypothesis=decoding_cfg.beam.get(
                    "return_best_hypothesis", True
                ),
                search_type="default",
                score_norm=self.cfg.beam.get("score_norm", True),
                softmax_temperature=self.cfg.beam.get("softmax_temperature", 1.0),
            )

        elif self.cfg.strategy == "maes":

            self.decoding = tdt_beam_decoding.BeamTDTInfer(
                decoder_model=decoder,
                joint_model=joint,
                durations=self.durations,
                beam_size=self.cfg.beam.beam_size,
                return_best_hypothesis=decoding_cfg.beam.get(
                    "return_best_hypothesis", True
                ),
                search_type="maes",
                score_norm=self.cfg.beam.get("score_norm", True),
                maes_num_steps=self.cfg.beam.get("maes_num_steps", 2),
                maes_prefix_alpha=self.cfg.beam.get("maes_prefix_alpha", 1),
                maes_expansion_gamma=self.cfg.beam.get("maes_expansion_gamma", 2.3),
                maes_expansion_beta=self.cfg.beam.get("maes_expansion_beta", 2),
                softmax_temperature=self.cfg.beam.get("softmax_temperature", 1.0),
                ngram_lm_model=self.cfg.beam.get("ngram_lm_model", None),
                ngram_lm_alpha=self.cfg.beam.get("ngram_lm_alpha", 0.3),
            )
        elif self.cfg.strategy == "malsd_batch":

            self.decoding = tdt_beam_decoding.BeamBatchedTDTInfer(
                decoder_model=decoder,
                joint_model=joint,
                blank_index=self.blank_id,
                durations=self.durations,
                beam_size=self.cfg.beam.beam_size,
                search_type="malsd_batch",
                max_symbols_per_step=self.cfg.beam.get("max_symbols", 10),
                ngram_lm_model=self.cfg.beam.get("ngram_lm_model", None),
                ngram_lm_alpha=self.cfg.beam.get("ngram_lm_alpha", 0.0),
                blank_lm_score_mode=self.cfg.beam.get(
                    "blank_lm_score_mode", BlankLMScoreMode.LM_WEIGHTED_FULL
                ),
                vocab_size=self.vocab_size,
                pruning_mode=self.cfg.beam.get("pruning_mode", PruningMode.LATE),
                score_norm=self.cfg.beam.get("score_norm", True),
                allow_cuda_graphs=self.cfg.beam.get("allow_cuda_graphs", True),
                return_best_hypothesis=self.cfg.beam.get(
                    "return_best_hypothesis", True
                ),
            )
        else:
            raise ValueError(
                f"Incorrect decoding strategy supplied. Must be one of {possible_strategies}\n"
                f"but was provided {self.cfg.strategy}"
            )

    def rnnt_decoder_predictions_tensor(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        return_hypotheses: bool = False,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
    ) -> Union[List[Hypothesis], List[List[Hypothesis]]]:
        """
        Decode an encoder output by autoregressive decoding of the Decoder+Joint networks.

        Args:
            encoder_output: torch.Tensor of shape [B, D, T].
            encoded_lengths: torch.Tensor containing lengths of the padded encoder outputs. Shape [B].
            return_hypotheses: bool. If set to True it will return list of Hypothesis or NBestHypotheses

        Returns:
            If `return_all_hypothesis` is set:
                A list[list[Hypothesis]].
                    Look at rnnt_utils.Hypothesis for more information.

            If `return_all_hypothesis` is not set:
                A list[Hypothesis].
                List of best hypotheses
                    Look at rnnt_utils.Hypothesis for more information.
        """
        # Compute hypotheses
        with torch.inference_mode():
            hypotheses_list = self.decoding(
                encoder_output=encoder_output,
                encoded_lengths=encoded_lengths,
                partial_hypotheses=partial_hypotheses,
            )  # type: [List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]

        prediction_list = hypotheses_list

        if isinstance(prediction_list[0], NBestHypotheses):
            hypotheses = []
            all_hypotheses = []

            for nbest_hyp in prediction_list:  # type: NBestHypotheses
                n_hyps = (
                    nbest_hyp.n_best_hypotheses
                )  # Extract all hypotheses for this sample
                decoded_hyps = self.decode_hypothesis(n_hyps)  # type: List[str]

                hypotheses.append(decoded_hyps[0])  # best hypothesis
                all_hypotheses.append(decoded_hyps)

            if return_hypotheses:
                return all_hypotheses  # type: list[list[Hypothesis]]

            all_hyp = [
                [Hypothesis(h.score, h.y_sequence, h.text) for h in hh]
                for hh in all_hypotheses
            ]
            return all_hyp

        else:
            hypotheses = self.decode_hypothesis(prediction_list)  # type: List[str]

            if return_hypotheses:
                # greedy decoding, can get high-level confidence scores
                if self.preserve_frame_confidence and (
                    self.preserve_word_confidence or self.preserve_token_confidence
                ):
                    hypotheses = self.compute_confidence(hypotheses)
                return hypotheses

            return [Hypothesis(h.score, h.y_sequence, h.text) for h in hypotheses]

    def decode_hypothesis(
        self, hypotheses_list: List[Hypothesis]
    ) -> List[Union[Hypothesis, NBestHypotheses]]:
        """
        Decode a list of hypotheses into a list of strings.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of strings.
        """
        for ind in range(len(hypotheses_list)):
            # Extract the integer encoded hypothesis
            prediction = hypotheses_list[ind].y_sequence

            if type(prediction) != list:
                prediction = prediction.tolist()

            # RNN-T sample level is already preprocessed by implicit RNNT decoding
            # Simply remove any blank and possibly big blank tokens

            if self._is_tdt:  # TDT model.
                prediction = [p for p in prediction if p < self.blank_id]
            else:  # standard RNN-T
                prediction = [p for p in prediction if p != self.blank_id]

            # De-tokenize the integer tokens; if not computing timestamps

            hypothesis = self.decode_tokens_to_str(prediction)

            # collapse leading spaces before . , ? for PC models
            hypothesis = re.sub(r"(\s+)([\.\,\?])", r"\2", hypothesis)

            # De-tokenize the integer tokens
            hypotheses_list[ind].text = hypothesis

        return hypotheses_list

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Decode a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = "".join(self.decode_ids_to_tokens(tokens))
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = [
            self.labels_map[c]
            for c in tokens
            if c < self.blank_id - self.num_extra_outputs
        ]
        return token_list
