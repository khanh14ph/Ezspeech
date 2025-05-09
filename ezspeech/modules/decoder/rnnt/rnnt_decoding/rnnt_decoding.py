
import copy
import re
import unicodedata
from abc import abstractmethod
from dataclasses import dataclass, field, is_dataclass
from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from ezspeech.modules.decoder.rnnt.rnnt_decoding import rnnt_beam_decoding, rnnt_greedy_decoding, tdt_beam_decoding
from ezspeech.modules.decoder.rnnt.rnnt_utils import  Hypothesis,NBestHypotheses
from ezspeech.modules.decoder.rnnt.rnnt_decoding.rnnt_batched_beam_utils import BlankLMScoreMode,PruningMode
class RNNTDecoding:
    """
    Used for performing RNN-T auto-regressive decoding of the Decoder+Joint network given the encoder state.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy, greedy_batch (for greedy decoding).
                -   beam, tsd, alsd (for beam search decoding).

            compute_hypothesis_token_set: A bool flag, which determines whether to compute a list of decoded
                tokens as well as the decoded string. Default is False in order to avoid double decoding
                unless required.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `alignments` in it. Here, `alignments` is a List of List of
                Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

                In order to obtain this hypothesis, please utilize `rnnt_decoder_predictions_tensor` function
                with the `return_hypotheses` flag set to True.

                The length of the list corresponds to the Acoustic Length (T).
                Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
                U is the number of target tokens for the current timestep Ti.

            tdt_include_token_duration: Bool flag, which determines whether predicted durations for each token
            need to be included in the Hypothesis object. Defaults to False.

            compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrete intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            rnnt_timestamp_type: A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps, "segment" for segment level time stamps and "all" (default), for character, word and
                segment level time stamps.

            word_seperator: Str token representing the seperator between words.

            segment_seperators: List containing tokens representing the seperator(s) between segments.

            segment_gap_threshold: The threshold (in frames) that caps the gap between two words necessary for forming
            the segments.

            preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                generated during decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `frame_confidence` in it. Here, `alignments` is a List of List of ints.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `rnnt_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `alignments` is a List of List of floats.

                    The length of the list corresponds to the Acoustic Length (T).
                    Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
                    U is the number of target tokens for the current timestep Ti.
                preserve_token_confidence: Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `token_confidence` in it. Here, `token_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized tokens.
                preserve_word_confidence: Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `word_confidence` in it. Here, `word_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized words.
                exclude_blank: Bool flag indicating that blank token confidence scores are to be excluded
                    from the `token_confidence`.
                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word
                    confidence. Valid options are `mean`, `min`, `max`, `prod`.
                tdt_include_duration: Bool flag indicating that the duration confidence scores are to be calculated and
                    attached to the regular frame confidence,
                    making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).
                method_cfg: A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name: The method name (str).
                        Supported values:
                            - 'max_prob' for using the maximum token probability as a confidence.
                            - 'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type: Which type of entropy to use (str).
                        Used if confidence_method_cfg.name is set to `entropy`.
                        Supported values:
                            - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                                the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                                Note that for this entropy, the alpha should comply the following inequality:
                                (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                                where V is the model vocabulary size.
                            - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                                Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/Tsallis_entropy
                            - 'renyi' for the Rényi entropy.
                                Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

                    alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the alpha equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm: A mapping of the entropy value to the interval [0,1].
                        Supported values:
                            - 'lin' for using the linear mapping.
                            - 'exp' for using exponential mapping with linear shift.

            The config may further contain the following sub-dictionaries:
            "greedy":
                max_symbols: int, describing the maximum number of target tokens to decode per
                    timestep during greedy decoding. Setting to larger values allows longer sentences
                    to be decoded, at the cost of increased execution time.
                preserve_frame_confidence: Same as above, overrides above value.
                confidence_method_cfg: Same as above, overrides confidence_cfg.method_cfg.

            "beam":
                beam_size: int, defining the beam size for beam search. Must be >= 1.
                    If beam_size == 1, will perform cached greedy search. This might be slightly different
                    results compared to the greedy search above.

                score_norm: optional bool, whether to normalize the returned beam score in the hypotheses.
                    Set to True by default.

                return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                    hypotheses after beam search has concluded. This flag is set by default.

                tsd_max_sym_exp: optional int, determines number of symmetric expansions of the target symbols
                    per timestep of the acoustic model. Larger values will allow longer sentences to be decoded,
                    at increased cost to execution time.

                alsd_max_target_len: optional int or float, determines the potential maximum target sequence length.
                    If an integer is provided, it can decode sequences of that particular maximum length.
                    If a float is provided, it can decode sequences of int(alsd_max_target_len * seq_len),
                    where seq_len is the length of the acoustic model output (T).

                    NOTE:
                        If a float is provided, it can be greater than 1!
                        By default, a float of 2.0 is used so that a target sequence can be at most twice
                        as long as the acoustic model output length T.

                maes_num_steps: Number of adaptive steps to take. From the paper, 2 steps is generally sufficient,
                    and can be reduced to 1 to improve decoding speed while sacrificing some accuracy. int > 0.

                maes_prefix_alpha: Maximum prefix length in prefix search. Must be an integer, and is advised to keep
                    this as 1 in order to reduce expensive beam search cost later. int >= 0.

                maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
                    Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
                    and affects the speed of inference since large values will perform large beam search in the next
                    step.

                maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the
                    expansions. The default (2.3) is selected from the paper. It performs a comparison
                    (max_log_prob - gamma <= log_prob[v]) where v is all vocabulary indices in the Vocab set and
                    max_log_prob is the "most" likely token to be predicted. Gamma therefore provides a margin of
                    additional tokens which can be potential candidates for expansion apart from the "most likely"
                    candidate. Lower values will reduce the number of expansions (by increasing pruning-by-value,
                    thereby improving speed but hurting accuracy). Higher values will increase the number of expansions
                    (by reducing pruning-by-value, thereby reducing speed but potentially improving accuracy). This is
                    a hyper parameter to be experimentally tuned on a validation set.

                softmax_temperature: Scales the logits of the joint prior to computing log_softmax.

        decoder: The Decoder/Prediction network module.
        joint: The Joint network module.
        vocabulary: The vocabulary (excluding the RNNT blank token) which will be used for decoding.
    """

    def __init__(
        self,
        decoding_cfg,
        decoder,
        joint,
        vocabulary,
    ):
        super(RNNTDecoding, self).__init__()

        # we need to ensure blank is the last token in the vocab for the case of RNNT and Multi-blank RNNT.
        blank_id = 0

        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

        # Convert dataclass to config object

        self.cfg = decoding_cfg
        self.blank_id = blank_id
        self.num_extra_outputs = joint.num_extra_outputs
        self.durations = self.cfg.get("durations", None)
        self.compute_hypothesis_token_set = self.cfg.get("compute_hypothesis_token_set", False)

        self.tdt_include_token_duration = self.cfg.get('tdt_include_token_duration', False)
        self.word_seperator = self.cfg.get('word_seperator', ' ')

        self._is_tdt = self.durations is not None and self.durations != []  # this means it's a TDT model.
        if self._is_tdt:
            if blank_id == 0:
                raise ValueError("blank_id must equal len(non_blank_vocabs) for TDT models")
            if self.durations is None:
                raise ValueError("duration can't not be None")
            if self.cfg.strategy not in ['greedy', 'greedy_batch', 'beam', 'maes', "malsd_batch"]:
                raise ValueError(
                    "currently only greedy, greedy_batch, beam and maes inference is supported for TDT models"
                )

        possible_strategies = ['greedy', 'greedy_batch', 'beam', 'alsd', 'maes', 'malsd_batch', "maes_batch"]
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}")

        # Update preserve alignments
        if self.preserve_alignments is None:
            if self.cfg.strategy in ['greedy', 'greedy_batch']:
                self.preserve_alignments = self.cfg.greedy.get('preserve_alignments', False)

            elif self.cfg.strategy in ['beam', 'tsd', 'alsd', 'maes']:
                self.preserve_alignments = self.cfg.beam.get('preserve_alignments', False)


        # initialize confidence-related field
        if self._is_tdt:
            self.tdt_include_token_duration = self.tdt_include_token_duration or self.compute_timestamps

    
        if self.cfg.strategy == 'greedy':

                self.decoding = rnnt_greedy_decoding.GreedyTDTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    durations=self.durations,
                    max_symbols_per_step=(
                        self.cfg.greedy.get('max_symbols', None)
                        or self.cfg.greedy.get('max_symbols_per_step', None)
                    ),
                    include_duration=self.tdt_include_token_duration,
                )

        elif self.cfg.strategy == 'greedy_batch':
                self.decoding = rnnt_greedy_decoding.GreedyBatchedTDTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    durations=self.durations,
                    max_symbols_per_step=(
                        self.cfg.greedy.get('max_symbols', None)
                        or self.cfg.greedy.get('max_symbols_per_step', None)
                    ),
                    include_duration=self.tdt_include_token_duration,

                    use_cuda_graph_decoder=self.cfg.greedy.get('use_cuda_graph_decoder', True),
                    ngram_lm_model=self.cfg.greedy.get('ngram_lm_model', None),
                    ngram_lm_alpha=self.cfg.greedy.get('ngram_lm_alpha', 0),
                )

        elif self.cfg.strategy == 'beam':
                self.decoding = tdt_beam_decoding.BeamTDTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    durations=self.durations,
                    beam_size=self.cfg.beam.beam_size,
                    return_best_hypothesis=decoding_cfg.beam.get('return_best_hypothesis', True),
                    search_type='default',
                    score_norm=self.cfg.beam.get('score_norm', True),
                    softmax_temperature=self.cfg.beam.get('softmax_temperature', 1.0),
                )

        



        elif self.cfg.strategy == 'maes':

            self.decoding = tdt_beam_decoding.BeamTDTInfer(
                decoder_model=decoder,
                joint_model=joint,
                durations=self.durations,
                beam_size=self.cfg.beam.beam_size,
                return_best_hypothesis=decoding_cfg.beam.get('return_best_hypothesis', True),
                search_type='maes',
                score_norm=self.cfg.beam.get('score_norm', True),
                maes_num_steps=self.cfg.beam.get('maes_num_steps', 2),
                maes_prefix_alpha=self.cfg.beam.get('maes_prefix_alpha', 1),
                maes_expansion_gamma=self.cfg.beam.get('maes_expansion_gamma', 2.3),
                maes_expansion_beta=self.cfg.beam.get('maes_expansion_beta', 2.0),
                softmax_temperature=self.cfg.beam.get('softmax_temperature', 1.0),

                ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 0.3),
            )
        elif self.cfg.strategy == 'malsd_batch':
            
            self.decoding = tdt_beam_decoding.BeamBatchedTDTInfer(
                decoder_model=decoder,
                joint_model=joint,
                blank_index=self.blank_id,
                durations=self.durations,
                beam_size=self.cfg.beam.beam_size,
                search_type='malsd_batch',
                max_symbols_per_step=self.cfg.beam.get("max_symbols", 10),
                ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 0.0),
                blank_lm_score_mode=self.cfg.beam.get(
                    'blank_lm_score_mode', BlankLMScoreMode.LM_WEIGHTED_FULL
                ),
                pruning_mode=self.cfg.beam.get('pruning_mode', PruningMode.LATE),
                score_norm=self.cfg.beam.get('score_norm', True),
                allow_cuda_graphs=self.cfg.beam.get('allow_cuda_graphs', True),
                return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
            )
        else:
            raise ValueError(
                f"Incorrect decoding strategy supplied. Must be one of {possible_strategies}\n"
                f"but was provided {self.cfg.strategy}"
            )


        
        if isinstance(self.decoding, rnnt_beam_decoding.BeamRNNTInfer) or isinstance(
            self.decoding, tdt_beam_decoding.BeamTDTInfer
        ):
            self.decoding.set_decoding_type('char')

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
            partial_hypotheses: Optional list of partial hypotheses to continue decoding from

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
                encoder_output=encoder_output, encoded_lengths=encoded_lengths, partial_hypotheses=partial_hypotheses
            )  # type: [List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]

        prediction_list = hypotheses_list

        if isinstance(prediction_list[0], NBestHypotheses):
            hypotheses = []
            all_hypotheses = []

            for nbest_hyp in prediction_list:  # type: NBestHypotheses
                n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
                decoded_hyps = self.decode_hypothesis(n_hyps)  # type: List[str]

                hypotheses.append(decoded_hyps[0])  # best hypothesis
                all_hypotheses.append(decoded_hyps)

            if return_hypotheses:
                return all_hypotheses  # type: list[list[Hypothesis]]

            all_hyp = [[Hypothesis(h.score, h.y_sequence, h.text) for h in hh] for hh in all_hypotheses]
            return all_hyp

        else:
            hypotheses = self.decode_hypothesis(prediction_list)  # type: List[str]

            # If computing timestamps
            if self.compute_timestamps is True:
                timestamp_type = self.cfg.get('rnnt_timestamp_type', 'all')
                for hyp_idx in range(len(hypotheses)):
                    hypotheses[hyp_idx] = self.compute_rnnt_timestamps(hypotheses[hyp_idx], timestamp_type)

            if return_hypotheses:
                # greedy decoding, can get high-level confidence scores
            
                return hypotheses

            return [Hypothesis(h.score, h.y_sequence, h.text) for h in hypotheses]

    def decode_hypothesis(self, hypotheses_list: List[Hypothesis]) -> List[Union[Hypothesis, NBestHypotheses]]:
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
            if self.big_blank_durations is not None and self.big_blank_durations != []:  # multi-blank RNNT
                num_extra_outputs = len(self.big_blank_durations)
                prediction = [p for p in prediction if p < self.blank_id - num_extra_outputs]
            elif self._is_tdt:  # TDT model.
                prediction = [p for p in prediction if p < self.blank_id]
            else:  # standard RNN-T
                prediction = [p for p in prediction if p != self.blank_id]

            # De-tokenize the integer tokens; if not computing timestamps
            if self.compute_timestamps is True and self._is_tdt:
                hypothesis = (prediction, None, None)
            elif self.compute_timestamps is True:
                # keep the original predictions, wrap with the number of repetitions per token and alignments
                # this is done so that `rnnt_decoder_predictions_tensor()` can process this hypothesis
                # in order to compute exact time stamps.
                alignments = copy.deepcopy(hypotheses_list[ind].alignments)
                token_repetitions = [1] * len(alignments)  # preserve number of repetitions per token
                hypothesis = (prediction, alignments, token_repetitions)
            else:
                hypothesis = self.decode_tokens_to_str(prediction)

                # collapse leading spaces before . , ? for PC models
                hypothesis = re.sub(r'(\s+)([\.\,\?])', r'\2', hypothesis)

                if self.compute_hypothesis_token_set:
                    hypotheses_list[ind].tokens = self.decode_ids_to_tokens(prediction)

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
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
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
        token_list = [self.labels_map[c] for c in tokens if c < self.blank_id - self.num_extra_outputs]
        return token_list






    