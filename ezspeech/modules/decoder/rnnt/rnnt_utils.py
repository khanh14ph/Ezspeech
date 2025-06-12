from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a packed torch.Tensor
        behaving in the same manner. dtype must be torch.Long in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    text: (Optional) A decoded string after processing via CTC / RNN-T decoding (removing the CTC/RNNT
        `blank` tokens, and optionally merging word-pieces). Should be used as decoded string for
        Word Error Rate calculation.

    timestamp: (Optional) A list of integer indices representing at which index in the decoding
        process did the token appear. Should be of same length as the number of non-blank tokens.

    alignments: (Optional) Represents the CTC / RNNT token alignments as integer tokens along an axis of
        time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of integer indices.
        For RNNT, represented as a dangling list of list of integer indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).
        The set of valid indices **includes** the CTC / RNNT blank token in order to represent alignments.

    frame_confidence: (Optional) Represents the CTC / RNNT per-frame confidence scores as token probabilities
        along an axis of time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of float indices.
        For RNNT, represented as a dangling list of list of float indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).

    token_confidence: (Optional) Represents the CTC / RNNT per-token confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    word_confidence: (Optional) Represents the CTC / RNNT per-word confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    length: Represents the length of the sequence (the original length without padding), otherwise
        defaults to 0.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.

    ngram_lm_state: (Optional) State of the external n-gram Language Model.

    tokens: (Optional) A list of decoded tokens (can be characters or word-pieces.

    last_token (Optional): A token or batch of tokens which was predicted in the last step.

    last_frame (Optional): Index of the last decoding step hypothesis was updated including blank token prediction.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestamp: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    frame_confidence: Optional[Union[List[float], List[List[float]]]] = None
    token_confidence: Optional[List[float]] = None
    word_confidence: Optional[List[float]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    ngram_lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None
    token_duration: Optional[torch.Tensor] = None
    last_frame: Optional[int] = None

    @property
    def non_blank_frame_confidence(self) -> List[float]:
        """Get per-frame confidence for non-blank tokens according to self.timestamp

        Returns:
            List with confidence scores. The length of the list is the same as `timestamp`.
        """
        non_blank_frame_confidence = []
        # self.timestamp can be a dict for RNNT
        timestamp = (
            self.timestamp["timestep"]
            if isinstance(self.timestamp, dict)
            else self.timestamp
        )
        if len(timestamp) != 0 and self.frame_confidence is not None:
            if any(isinstance(i, list) for i in self.frame_confidence):  # rnnt
                t_prev = -1
                offset = 0
                for t in timestamp:
                    if t != t_prev:
                        t_prev = t
                        offset = 0
                    else:
                        offset += 1
                    non_blank_frame_confidence.append(self.frame_confidence[t][offset])
            else:  # ctc
                non_blank_frame_confidence = [
                    self.frame_confidence[t] for t in timestamp
                ]
        return non_blank_frame_confidence

    @property
    def words(self) -> List[str]:
        """Get words from self.text

        Returns:
            List with words (str).
        """
        return [] if self.text is None else self.text.split()


@dataclass
class NBestHypotheses:
    """List of N best hypotheses"""

    n_best_hypotheses: Optional[List[Hypothesis]]


def is_prefix(x: List[int], pref: List[int]) -> bool:
    """
    Obtained from https://github.com/espnet/espnet.

    Check if pref is a prefix of x.

    Args:
        x: Label ID sequence.
        pref: Prefix label ID sequence.

    Returns:
        : Whether pref is a prefix of x.
    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False

    return True


class ConfidenceMethodConstants:
    NAMES = ("max_prob", "entropy")
    ENTROPY_TYPES = ("gibbs", "tsallis", "renyi")
    ENTROPY_NORMS = ("lin", "exp")

    @classmethod
    def print(cls):
        return (
            cls.__name__
            + ": "
            + str(
                {
                    "NAMES": cls.NAMES,
                    "ENTROPY_TYPES": cls.ENTROPY_TYPES,
                    "ENTROPY_NORMS": cls.ENTROPY_NORMS,
                }
            )
        )


class ConfidenceMethodConfig:
    """A Config which contains the method name and settings to compute per-frame confidence scores.

    Args:
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
    """

    name: str = "entropy"
    entropy_type: str = "tsallis"
    alpha: float = 0.33
    entropy_norm: str = "exp"
    temperature: str = "DEPRECATED"

    def __post_init__(self):
        if self.temperature != "DEPRECATED":
            # self.temperature has type str
            self.alpha = float(self.temperature)
            self.temperature = "DEPRECATED"
        if self.name not in ConfidenceMethodConstants.NAMES:
            raise ValueError(
                f"`name` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.NAMES) + '`'}. Provided: `{self.name}`"
            )
        if self.entropy_type not in ConfidenceMethodConstants.ENTROPY_TYPES:
            raise ValueError(
                f"`entropy_type` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.ENTROPY_TYPES) + '`'}. Provided: `{self.entropy_type}`"
            )
        if self.alpha <= 0.0:
            raise ValueError(f"`alpha` must be > 0. Provided: {self.alpha}")
        if self.entropy_norm not in ConfidenceMethodConstants.ENTROPY_NORMS:
            raise ValueError(
                f"`entropy_norm` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.ENTROPY_NORMS) + '`'}. Provided: `{self.entropy_norm}`"
            )


class BatchedHyps:
    """Class to store batched hypotheses (labels, time_indices, scores) for efficient RNNT decoding"""

    def __init__(
        self,
        batch_size: int,
        init_length: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
    ):
        """

        Args:
            batch_size: batch size for hypotheses
            init_length: initial estimate for the length of hypotheses (if the real length is higher,
                tensors will be reallocated)
            device: device for storing hypotheses
            float_dtype: float type for scores
        """
        if init_length <= 0:
            raise ValueError(f"init_length must be > 0, got {init_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self._max_length = init_length

        # batch of current lengths of hypotheses and correspoinding timestamps
        self.current_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        # tensor for storing transcripts
        self.transcript = torch.zeros(
            (batch_size, self._max_length), device=device, dtype=torch.long
        )
        # tensor for storing timestamps corresponding to transcripts
        self.timestamps = torch.zeros(
            (batch_size, self._max_length), device=device, dtype=torch.long
        )
        # tensor for storing durations corresponding to transcripts tokens
        self.token_durations = torch.zeros(
            (batch_size, self._max_length), device=device, dtype=torch.long
        )
        # accumulated scores for hypotheses
        self.scores = torch.zeros(batch_size, device=device, dtype=float_dtype)

        # tracking last timestamp of each hyp to avoid infinite looping (when max symbols per frame is restricted)
        # last observed timestamp (with label) for each hypothesis
        self.last_timestamp = torch.full(
            (batch_size,), -1, device=device, dtype=torch.long
        )
        # number of labels for the last timestamp
        self.last_timestamp_lasts = torch.zeros(
            batch_size, device=device, dtype=torch.long
        )
        self._batch_indices = torch.arange(batch_size, device=device)
        self._ones_batch = torch.ones_like(self._batch_indices)

    def clear_(self):
        """
        Clears batched hypotheses state.
        """
        self.current_lengths.fill_(0)
        self.transcript.fill_(0)
        self.timestamps.fill_(0)
        self.token_durations.fill_(0)
        self.scores.fill_(0.0)
        self.last_timestamp.fill_(-1)
        self.last_timestamp_lasts.fill_(0)

    def _allocate_more(self):
        """
        Allocate 2x space for tensors, similar to common C++ std::vector implementations
        to maintain O(1) insertion time complexity
        """
        self.transcript = torch.cat(
            (self.transcript, torch.zeros_like(self.transcript)), dim=-1
        )
        self.timestamps = torch.cat(
            (self.timestamps, torch.zeros_like(self.timestamps)), dim=-1
        )
        self.token_durations = torch.cat(
            (self.token_durations, torch.zeros_like(self.token_durations)), dim=-1
        )
        self._max_length *= 2

    def add_results_(
        self,
        active_indices: torch.Tensor,
        labels: torch.Tensor,
        time_indices: torch.Tensor,
        scores: torch.Tensor,
        token_durations: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Args:
            active_indices: tensor with indices of active hypotheses (indices should be within the original batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
        """
        if active_indices.shape[0] == 0:
            return  # nothing to add
        # if needed - increase storage
        if self.current_lengths.max().item() >= self._max_length:
            self._allocate_more()

        self.add_results_no_checks_(
            active_indices=active_indices,
            labels=labels,
            time_indices=time_indices,
            scores=scores,
            token_durations=token_durations,
        )

    def add_results_no_checks_(
        self,
        active_indices: torch.Tensor,
        labels: torch.Tensor,
        time_indices: torch.Tensor,
        scores: torch.Tensor,
        token_durations: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses without checks.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Useful if all the memory is pre-allocated, especially with cuda graphs
        (otherwise prefer a more safe `add_results_`)
        Args:
            active_indices: tensor with indices of active hypotheses (indices should be within the original batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
            token_durations: predicted durations for each token by TDT head
        """
        # accumulate scores
        self.scores[active_indices] += scores

        # store transcript and timestamps
        active_lengths = self.current_lengths[active_indices]
        self.transcript[active_indices, active_lengths] = labels
        self.timestamps[active_indices, active_lengths] = time_indices
        if token_durations is not None:
            self.token_durations[active_indices, active_lengths] = token_durations
        # store last observed timestamp + number of observation for the current timestamp
        self.last_timestamp_lasts[active_indices] = torch.where(
            self.last_timestamp[active_indices] == time_indices,
            self.last_timestamp_lasts[active_indices] + 1,
            1,
        )
        self.last_timestamp[active_indices] = time_indices
        # increase lengths
        self.current_lengths[active_indices] += 1

    def add_results_masked_(
        self,
        active_mask: torch.Tensor,
        labels: torch.Tensor,
        time_indices: torch.Tensor,
        scores: torch.Tensor,
        token_durations: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Args:
            active_mask: tensor with mask for active hypotheses (of batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
        """
        if (self.current_lengths + active_mask).max() >= self._max_length:
            self._allocate_more()
        self.add_results_masked_no_checks_(
            active_mask=active_mask,
            labels=labels,
            time_indices=time_indices,
            scores=scores,
            token_durations=token_durations,
        )

    def add_results_masked_no_checks_(
        self,
        active_mask: torch.Tensor,
        labels: torch.Tensor,
        time_indices: torch.Tensor,
        scores: torch.Tensor,
        token_durations: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses without checks.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Useful if all the memory is pre-allocated, especially with cuda graphs
        (otherwise prefer a more safe `add_results_`)
        Args:
            active_mask: tensor with mask for active hypotheses (of batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
        """
        # accumulate scores
        # same as self.scores[active_mask] += scores[active_mask], but non-blocking
        torch.where(active_mask, self.scores + scores, self.scores, out=self.scores)

        # store transcript and timestamps
        self.transcript[self._batch_indices, self.current_lengths] = labels
        self.timestamps[self._batch_indices, self.current_lengths] = time_indices
        if token_durations is not None:
            self.token_durations[self._batch_indices, self.current_lengths] = (
                token_durations
            )
        # store last observed timestamp + number of observation for the current timestamp
        # if last_timestamp == time_indices, increase; else set to 1
        torch.where(
            torch.logical_and(active_mask, self.last_timestamp == time_indices),
            self.last_timestamp_lasts + 1,
            self.last_timestamp_lasts,
            out=self.last_timestamp_lasts,
        )
        torch.where(
            torch.logical_and(active_mask, self.last_timestamp != time_indices),
            self._ones_batch,
            self.last_timestamp_lasts,
            out=self.last_timestamp_lasts,
        )
        # same as: self.last_timestamp[active_mask] = time_indices[active_mask], but non-blocking
        torch.where(
            active_mask, time_indices, self.last_timestamp, out=self.last_timestamp
        )
        # increase lengths
        self.current_lengths += active_mask


class BatchedAlignments:
    """
    Class to store batched alignments (logits, labels, frame_confidence).
    Size is different from hypotheses, since blank outputs are preserved
    """

    def __init__(
        self,
        batch_size: int,
        logits_dim: int,
        init_length: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
        store_alignments: bool = True,
        store_frame_confidence: bool = False,
        with_duration_confidence: bool = False,
    ):
        """

        Args:
            batch_size: batch size for hypotheses
            logits_dim: dimension for logits
            init_length: initial estimate for the lengths of flatten alignments
            device: device for storing data
            float_dtype: expected logits/confidence data type
            store_alignments: if alignments should be stored
            store_frame_confidence: if frame confidence should be stored
        """
        if init_length <= 0:
            raise ValueError(f"init_length must be > 0, got {init_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.with_frame_confidence = store_frame_confidence
        self.with_duration_confidence = with_duration_confidence
        self.with_alignments = store_alignments
        self._max_length = init_length

        # tensor to store observed timestamps (for alignments / confidence scores)
        self.timestamps = torch.zeros(
            (batch_size, self._max_length), device=device, dtype=torch.long
        )
        # current lengths of the utterances (alignments)
        self.current_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)

        # empty tensors instead of None to make torch.jit.script happy
        self.logits = torch.zeros(0, device=device, dtype=float_dtype)
        self.labels = torch.zeros(0, device=device, dtype=torch.long)
        if self.with_alignments:
            # logits and labels; labels can contain <blank>, different from BatchedHyps
            self.logits = torch.zeros(
                (batch_size, self._max_length, logits_dim),
                device=device,
                dtype=float_dtype,
            )
            self.labels = torch.zeros(
                (batch_size, self._max_length), device=device, dtype=torch.long
            )

        # empty tensor instead of None to make torch.jit.script happy
        self.frame_confidence = torch.zeros(0, device=device, dtype=float_dtype)
        if self.with_frame_confidence:
            # tensor to store frame confidence
            self.frame_confidence = torch.zeros(
                (
                    [batch_size, self._max_length, 2]
                    if self.with_duration_confidence
                    else [batch_size, self._max_length]
                ),
                device=device,
                dtype=float_dtype,
            )
        self._batch_indices = torch.arange(batch_size, device=device)

    def clear_(self):
        """
        Clears batched hypotheses state.
        """
        self.current_lengths.fill_(0)
        self.timestamps.fill_(0)
        self.logits.fill_(0.0)
        self.labels.fill_(0)
        self.frame_confidence.fill_(0)

    def _allocate_more(self):
        """
        Allocate 2x space for tensors, similar to common C++ std::vector implementations
        to maintain O(1) insertion time complexity
        """
        self.timestamps = torch.cat(
            (self.timestamps, torch.zeros_like(self.timestamps)), dim=-1
        )
        if self.with_alignments:
            self.logits = torch.cat((self.logits, torch.zeros_like(self.logits)), dim=1)
            self.labels = torch.cat(
                (self.labels, torch.zeros_like(self.labels)), dim=-1
            )
        if self.with_frame_confidence:
            self.frame_confidence = torch.cat(
                (self.frame_confidence, torch.zeros_like(self.frame_confidence)), dim=1
            )
        self._max_length *= 2

    def add_results_(
        self,
        active_indices: torch.Tensor,
        time_indices: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        All tensors must use the same fixed batch dimension.
        Args:
            active_mask: tensor with mask for active hypotheses (of batch_size)
            logits: tensor with raw network outputs
            labels: tensor with decoded labels (can contain blank)
            time_indices: tensor of time index for each label
            confidence: optional tensor with confidence for each item in batch
        """
        # we assume that all tensors have the same first dimension
        if active_indices.shape[0] == 0:
            return  # nothing to add

        # if needed - increase storage
        if self.current_lengths.max().item() >= self._max_length:
            self._allocate_more()

        active_lengths = self.current_lengths[active_indices]
        # store timestamps - same for alignments / confidence
        self.timestamps[active_indices, active_lengths] = time_indices

        if self.with_alignments and logits is not None and labels is not None:
            self.logits[active_indices, active_lengths] = logits
            self.labels[active_indices, active_lengths] = labels

        if self.with_frame_confidence and confidence is not None:
            self.frame_confidence[active_indices, active_lengths] = confidence
        # increase lengths
        self.current_lengths[active_indices] += 1

    def add_results_masked_(
        self,
        active_mask: torch.Tensor,
        time_indices: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        All tensors must use the same fixed batch dimension.
        Args:
            active_mask: tensor with indices of active hypotheses (indices should be within the original batch_size)
            time_indices: tensor of time index for each label
            logits: tensor with raw network outputs
            labels: tensor with decoded labels (can contain blank)
            confidence: optional tensor with confidence for each item in batch
        """
        if (self.current_lengths + active_mask).max() >= self._max_length:
            self._allocate_more()
        self.add_results_masked_no_checks_(
            active_mask=active_mask,
            time_indices=time_indices,
            logits=logits,
            labels=labels,
            confidence=confidence,
        )

    def add_results_masked_no_checks_(
        self,
        active_mask: torch.Tensor,
        time_indices: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        All tensors must use the same fixed batch dimension.
        Useful if all the memory is pre-allocated, especially with cuda graphs
        (otherwise prefer a more safe `add_results_masked_`)
        Args:
            active_mask: tensor with indices of active hypotheses (indices should be within the original batch_size)
            time_indices: tensor of time index for each label
            logits: tensor with raw network outputs
            labels: tensor with decoded labels (can contain blank)
            confidence: optional tensor with confidence for each item in batch
        """
        # store timestamps - same for alignments / confidence
        self.timestamps[self._batch_indices, self.current_lengths] = time_indices

        if self.with_alignments and logits is not None and labels is not None:
            self.timestamps[self._batch_indices, self.current_lengths] = time_indices
            self.logits[self._batch_indices, self.current_lengths] = logits
            self.labels[self._batch_indices, self.current_lengths] = labels

        if self.with_frame_confidence and confidence is not None:
            self.frame_confidence[self._batch_indices, self.current_lengths] = (
                confidence
            )
        # increase lengths
        self.current_lengths += active_mask


def batched_hyps_to_hypotheses(
    batched_hyps: BatchedHyps,
    alignments: Optional[BatchedAlignments] = None,
    batch_size=None,
) -> List[Hypothesis]:
    """
    Convert batched hypotheses to a list of Hypothesis objects.
    Keep this function separate to allow for jit compilation for BatchedHyps class (see tests)

    Args:
        batched_hyps: BatchedHyps object
        alignments: BatchedAlignments object, optional; must correspond to BatchedHyps if present
        batch_size: Batch Size to retrieve hypotheses. When working with CUDA graphs the batch size for all tensors
            is constant, thus we need here the real batch size to return only necessary hypotheses

    Returns:
        list of Hypothesis objects
    """
    assert batch_size is None or batch_size <= batched_hyps.scores.shape[0]
    num_hyps = batched_hyps.scores.shape[0] if batch_size is None else batch_size
    # We clone `timestamps` and `y_sequence` to avoid keeping references
    # to the tensors that can be allocated by CUDA graphs. If we don't do this,
    # subsequent batches might reuse and overwrite the same memory,
    # leading to unexpected results due to shared references.
    hypotheses = [
        Hypothesis(
            score=batched_hyps.scores[i].item(),
            y_sequence=batched_hyps.transcript[
                i, : batched_hyps.current_lengths[i]
            ].clone(),
            timestamp=batched_hyps.timestamps[
                i, : batched_hyps.current_lengths[i]
            ].clone(),
            token_duration=(
                durations
                if not torch.all(
                    (
                        durations := batched_hyps.token_durations[
                            i, : batched_hyps.current_lengths[i]
                        ]
                    )
                    == 0
                )
                else torch.empty(0)
            ),
            alignments=None,
            dec_state=None,
        )
        for i in range(num_hyps)
    ]
    if alignments is not None:
        # move all data to cpu to avoid overhead with moving data by chunks
        alignment_lengths = alignments.current_lengths.cpu().tolist()
        if alignments.with_alignments:
            alignment_logits = alignments.logits.cpu()
            alignment_labels = alignments.labels.cpu()
        if alignments.with_frame_confidence:
            frame_confidence = alignments.frame_confidence.cpu()

        # for each hypothesis - aggregate alignment using unique_consecutive for time indices (~itertools.groupby)
        for i in range(len(hypotheses)):
            hypotheses[i].alignments = []
            if alignments.with_frame_confidence:
                hypotheses[i].frame_confidence = []
            _, grouped_counts = torch.unique_consecutive(
                alignments.timestamps[i, : alignment_lengths[i]], return_counts=True
            )
            start = 0
            for timestamp_cnt in grouped_counts.tolist():
                if alignments.with_alignments:
                    hypotheses[i].alignments.append(
                        [
                            (
                                alignment_logits[i, start + j],
                                alignment_labels[i, start + j],
                            )
                            for j in range(timestamp_cnt)
                        ]
                    )
                if alignments.with_frame_confidence:
                    hypotheses[i].frame_confidence.append(
                        [frame_confidence[i, start + j] for j in range(timestamp_cnt)]
                    )
                start += timestamp_cnt
    return hypotheses
