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
