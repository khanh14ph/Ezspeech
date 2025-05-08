from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from ezspeech.modules.decoder.rnnt.rnnt_utils import ConfidenceMethodConfig
from ezspeech.modules.decoder.rnnt.rnnt_utils import Hypothesis
from ezspeech.modules.decoder.rnnt.rnn import label_collate
from ezspeech.modules.decoder.rnnt.rnnt import RNNTDecoder, RNNTJoint
from ezspeech.modules.decoder.rnnt.rnnt_decoding.confidence_utils import ConfidenceMethodMixin
def pack_hypotheses(
    hypotheses: List[Hypothesis],
    logitlen: torch.Tensor,
) -> List[Hypothesis]:
    """
    Packs a list of hypotheses into a tensor and prepares decoder states.

    This function takes a list of token sequences (hypotheses) and converts
    it into a tensor format. If any decoder states are on the GPU, they
    are moved to the CPU. Additionally, the function removes any timesteps
    with a value of -1 from the sequences.

    Args:
        hypotheses (list): A list of token sequences representing hypotheses.

    Returns:
        list: A list of packed hypotheses in tensor format.
    """
    if hasattr(logitlen, 'cpu'):
        logitlen_cpu = logitlen.to('cpu')
    else:
        logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):  # type: rnnt_utils.Hypothesis
        hyp.y_sequence = (
            hyp.y_sequence.to(torch.long)
            if isinstance(hyp.y_sequence, torch.Tensor)
            else torch.tensor(hyp.y_sequence, dtype=torch.long)
        )
        hyp.length = logitlen_cpu[idx]

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

    return hypotheses

def _states_to_device(dec_state, device='cpu'):
    """
    Transfers decoder states to the specified device.

    This function moves the provided decoder states to the specified device (e.g., 'cpu' or 'cuda').

    Args:
        dec_state (Tensor): The decoder states to be transferred.
        device (str): The target device to which the decoder states should be moved. Defaults to 'cpu'.

    Returns:
        Tensor: The decoder states on the specified device.
    """
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class _GreedyRNNTInfer(ConfidenceMethodMixin):
    """A greedy transducer decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit
        confidence_method_cfg: A dict-like object which contains the method name and settings to compute per-frame
            confidence scores.

            name: The method name (str).
                Supported values:
                    - 'max_prob' for using the maximum token probability as a confidence.
                    - 'entropy' for using a normalized entropy of a log-likelihood vector.

            entropy_type: Which type of entropy to use (str). Used if confidence_method_cfg.name is set to `entropy`.
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

   
    def __init__(
        self,
        decoder_model: RNNTDecoder,
        joint_model: RNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = blank_index  # Start of single index

        if max_symbols_per_step is not None and max_symbols_per_step <= 0:
            raise ValueError(f"Expected max_symbols_per_step > 0 (or None), got {max_symbols_per_step}")
        self.max_symbols = max_symbols_per_step

        # set confidence calculation method
        self._init_confidence_method(confidence_method_cfg)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden: (Optional torch.Tensor): RNN State vector
            add_sos (bool): Whether to add a zero vector at the begging as "start of sentence" token.
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, U, H) if add_sos is false, else (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            if label.dtype != torch.long:
                label = label.long()

        else:
            # Label is an integer
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=add_sos, batch_size=batch_size)

            label = label_collate([[label]])

        # output: [B, 1, K]
        return self.decoder.predict(label, hidden, add_sos=add_sos, batch_size=batch_size)

    def _joint_step(self, enc, pred, log_normalize: Optional[bool] = None):
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            logits = self.joint.joint(enc, pred)

            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits

    def _joint_step_after_projection(self, enc, pred, log_normalize: Optional[bool] = None) -> torch.Tensor:
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model after projection. A torch.Tensor of shape [B, 1, H]
            pred: Output of the Decoder model after projection. A torch.Tensor of shape [B, 1, H]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            logits = self.joint.joint_after_projection(enc, pred)

            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits
class GreedyBatchedRNNTInferConfig:
    """Greedy Batched RNNT Infer Config"""

    max_symbols_per_step: Optional[int] = 10
    tdt_include_token_duration: bool = False
    tdt_include_duration_confidence: bool = False
    confidence_method_cfg: Optional[ConfidenceMethodConfig] = field(default_factory=lambda: ConfidenceMethodConfig())
    loop_labels: bool = True
    use_cuda_graph_decoder: bool = True
    ngram_lm_model: Optional[str] = None
    ngram_lm_alpha: float = 0.0

    def __post_init__(self):
        # OmegaConf.structured ensures that post_init check is always executed
        self.confidence_method_cfg = OmegaConf.structured(
            self.confidence_method_cfg
            if isinstance(self.confidence_method_cfg, ConfidenceMethodConfig)
            else ConfidenceMethodConfig(**self.confidence_method_cfg)
        )
class GreedyTDTInfer(_GreedyRNNTInfer):
    """A greedy TDT decoder.

    Sequence level greedy decoding, performed auto-regressively.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Must be len(vocabulary) for TDT models.
        durations: a list containing durations for TDT.
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        include_duration: Bool flag, which determines whether predicted durations for each token
            need to be included in the Hypothesis object. Defaults to False.
        include_duration_confidence: Bool flag indicating that the duration confidence scores are to be calculated and
            attached to the regular frame confidence,
            making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).
        confidence_method_cfg: A dict-like object which contains the method name and settings to compute per-frame
            confidence scores.

            name: The method name (str).
                Supported values:
                    - 'max_prob' for using the maximum token probability as a confidence.
                    - 'entropy' for using a normalized entropy of a log-likelihood vector.

            entropy_type: Which type of entropy to use (str). Used if confidence_method_cfg.name is set to `entropy`.
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

    def __init__(
        self,
        decoder_model: RNNTDecoder,
        joint_model: RNNTJoint,
        blank_index: int,
        durations: list,
        max_symbols_per_step: Optional[int] = None,
     
        include_duration: bool = False,
        include_duration_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            confidence_method_cfg=confidence_method_cfg,
        )
        self.durations = durations
        self.include_duration = include_duration
        self.include_duration_confidence = include_duration_confidence

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-regressively.
        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.
        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.inference_mode():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)

            self.decoder.eval()
            self.joint.eval()

            hypotheses = []
            # Process each sequence independently
            for batch_idx in range(encoder_output.size(0)):
                inseq = encoder_output[batch_idx, :, :].unsqueeze(1)  # [T, 1, D]
                logitlen = encoded_lengths[batch_idx]

                partial_hypothesis = partial_hypotheses[batch_idx] if partial_hypotheses is not None else None
                hypothesis = self._greedy_decode(inseq, logitlen, partial_hypotheses=partial_hypothesis)
                hypotheses.append(hypothesis)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypotheses, encoded_lengths)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode(
        self, x: torch.Tensor, out_len: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ):
        # x: [T, 1, D]
        # out_len: [seq_len]

        # Initialize blank state and empty label set in Hypothesis
        hypothesis = Hypothesis(
            score=0.0, y_sequence=[], dec_state=None, timestamp=[], token_duration=[], last_token=None
        )

        if partial_hypotheses is not None:
            hypothesis.last_token = partial_hypotheses.last_token
            hypothesis.y_sequence = (
                partial_hypotheses.y_sequence.cpu().tolist()
                if isinstance(partial_hypotheses.y_sequence, torch.Tensor)
                else partial_hypotheses.y_sequence
            )
            if partial_hypotheses.dec_state is not None:
                hypothesis.dec_state = self.decoder.batch_concat_states([partial_hypotheses.dec_state])
                hypothesis.dec_state = _states_to_device(hypothesis.dec_state, x.device)



        time_idx = 0
        while time_idx < out_len:
            # Extract encoder embedding at timestep t
            # f = x[time_idx, :, :].unsqueeze(0)  # [1, 1, D]
            f = x.narrow(dim=0, start=time_idx, length=1)

            # Setup exit flags and counter
            symbols_added = 0

            need_loop = True
            # While blank is not predicted, or we dont run out of max symbols per timestep
            while need_loop and (self.max_symbols is None or symbols_added < self.max_symbols):
                # In the first timestep, we initialize the network with RNNT Blank
                # In later timesteps, we provide previous predicted label as input.
                if hypothesis.last_token is None and hypothesis.dec_state is None:
                    last_label = self._SOS
                else:
                    last_label = label_collate([[hypothesis.last_token]])

                # Perform prediction network and joint network steps.
                g, hidden_prime = self._pred_step(last_label, hypothesis.dec_state)
                # If preserving per-frame confidence, log_normalize must be true
                logits = self._joint_step(f, g, log_normalize=False)
                logp = logits[0, 0, 0, : -len(self.durations)]
                duration_logp = torch.log_softmax(logits[0, 0, 0, -len(self.durations) :], dim=-1)
                del g

                # torch.max(0) op doesnt exist for FP 16.
                if logp.dtype != torch.float32:
                    logp = logp.float()

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()  # K is the label at timestep t_s in inner loop, s >= 0.

                d_v, d_k = duration_logp.max(0)
                d_k = d_k.item()

                skip = self.durations[d_k]

          

            

                del logp

                # If blank token is predicted, exit inner loop, move onto next timestep t
                if k != self._blank_index:
                    # Append token to label set, update RNN state.
                    hypothesis.y_sequence.append(k)
                    hypothesis.score += float(v)
                    hypothesis.timestamp.append(time_idx)
                    hypothesis.dec_state = hidden_prime
                    hypothesis.last_token = k
                    if self.include_duration:
                        hypothesis.token_duration.append(skip)

                # Increment token counter.
                symbols_added += 1
                time_idx += skip
                need_loop = skip == 0

            # this rarely happens, but we manually increment the `skip` number
            # if blank is emitted and duration=0 is predicted. This prevents possible
            # infinite loops.
            if skip == 0:
                skip = 1


            if symbols_added == self.max_symbols:
                time_idx += 1

      

        # Unpack the hidden states
        hypothesis.dec_state = self.decoder.batch_select_state(hypothesis.dec_state, 0)

        return hypothesis
