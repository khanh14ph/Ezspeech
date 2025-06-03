from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig

from ezspeech.modules.decoder.rnnt import rnnt_utils
from ezspeech.modules.decoder.rnnt import rnn


class RNNTDecoder(torch.nn.Module):
    """A Recurrent Neural Network Transducer Decoder / Prediction Network (RNN-T Prediction Network).
    An RNN-T Decoder/Prediction network, comprised of a stateful LSTM model.

    Args:
        prednet: A dict-like object which contains the following key-value pairs.

            pred_hidden:
                int specifying the hidden dimension of the prediction net.

            pred_rnn_layers:
                int specifying the number of rnn layers.

            Optionally, it may also contain the following:

                forget_gate_bias:
                    float, set by default to 1.0, which constructs a forget gate
                    initialized to 1.0.
                    Reference:
                    [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

                t_max:
                    int value, set to None by default. If an int is specified, performs Chrono Initialization
                    of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                    of training.
                    Reference:
                    [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

                weights_init_scale:
                    Float scale of the weights after initialization. Setting to lower than one
                    sometimes helps reduce variance between runs.

                hidden_hidden_bias_scale:
                    Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                    the default behaviour.

                dropout:
                    float, set to 0.0 by default. Optional dropout applied at the end of the final LSTM RNN layer.

        vocab_size: int, specifying the vocabulary size of the embedding layer of the Prediction network,
            excluding the RNNT blank token.

        normalization_mode: Can be either None, 'batch' or 'layer'. By default, is set to None.
            Defines the type of normalization applied to the RNN layer.

        random_state_sampling: bool, set to False by default. When set, provides normal-distribution
            sampled state tensors instead of zero tensors during training.
            Reference:
            [Recognizing long-form speech using streaming end-to-end models](https://arxiv.org/abs/1910.11455)

        blank_as_pad: bool, set to True by default. When set, will add a token to the Embedding layer of this
            prediction network, and will treat this token as a pad token. In essence, the RNNT pad token will
            be treated as a pad token, and the embedding layer will return a zero tensor for this token.

            It is set by default as it enables various batch optimizations required for batched beam search.
            Therefore, it is not recommended to disable this flag.
    """

    def __init__(
        self,
        prednet: Dict[str, Any],
        vocab_size: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        blank_as_pad: bool = True,
        blank_idx=0
    ):
        # Required arguments
        super().__init__()
        self.pred_hidden = prednet["pred_hidden"]
        self.pred_rnn_layers = prednet["pred_rnn_layers"]
        self.blank_idx = blank_idx
        self.blank_as_pad = blank_as_pad
        self.vocab_size=vocab_size
        # Initialize the model (blank token increases vocab size by 1)

        # Optional arguments
        forget_gate_bias = prednet.get("forget_gate_bias", 1.0)
        t_max = prednet.get("t_max", None)
        weights_init_scale = prednet.get("weights_init_scale", 1.0)
        hidden_hidden_bias_scale = prednet.get("hidden_hidden_bias_scale", 0.0)
        dropout = prednet.get("dropout", 0.0)
        self.random_state_sampling = random_state_sampling

        self.prediction = self._predict_modules(
            vocab_size=vocab_size,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            pred_rnn_layers=self.pred_rnn_layers,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            norm=normalization_mode,
            weights_init_scale=weights_init_scale,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            dropout=dropout,
            rnn_hidden_size=prednet.get("rnn_hidden_size", -1),
        )
        self._rnnt_export = False

    def forward(self, targets, target_length, states=None):
        # y: (B, U)
        y = rnn.label_collate(targets)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        if self._rnnt_export:
            add_sos = False
        else:
            add_sos = True

        g, states = self.predict(y, state=states, add_sos=add_sos)  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, target_length, states

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Stateful prediction of scores and state for a (possibly null) tokenset.
        This method takes various cases into consideration :
        - No token, no state - used for priming the RNN
        - No token, state provided - used for blank token scoring
        - Given token, states - used for scores + new states

        Here:
        B - batch size
        U - label length
        H - Hidden dimension size of RNN
        L - Number of RNN layers

        Args:
            y: Optional torch tensor of shape [B, U] of dtype long which will be passed to the Embedding.
                If None, creates a zero tensor of shape [B, 1, H] which mimics output of pad-token on EmbeddiNg.

            state: An optional list of states for the RNN. Eg: For LSTM, it is the state list length is 2.
                Each state must be a tensor of shape [L, B, H].
                If None, and during training mode and `random_state_sampling` is set, will sample a
                normal distribution tensor of the above shape. Otherwise, None will be passed to the RNN.

            add_sos: bool flag, whether a zero vector describing a "start of signal" token should be
                prepended to the above "y" tensor. When set, output size is (B, U + 1, H).

            batch_size: An optional int, specifying the batch size of the `y` tensor.
                Can be infered if `y` and `state` is None. But if both are None, then batch_size cannot be None.

        Returns:
            A tuple  (g, hid) such that -

            If add_sos is False:

                g:
                    (B, U, H)

                hid:
                    (h, c) where h is the final sequence hidden state and c is the final cell state:

                        h (tensor), shape (L, B, H)

                        c (tensor), shape (L, B, H)

            If add_sos is True:
                g:
                    (B, U + 1, H)

                hid:
                    (h, c) where h is the final sequence hidden state and c is the final cell state:

                        h (tensor), shape (L, B, H)

                        c (tensor), shape (L, B, H)

        """
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        # If in training mode, and random_state_sampling is set,
        # initialize state to random normal distribution tensor.
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)

        # Forward step through RNN
        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state

        return g, hid

    def _predict_modules(
        self,
        vocab_size,
        pred_n_hidden,
        pred_rnn_layers,
        forget_gate_bias,
        t_max,
        norm,
        weights_init_scale,
        hidden_hidden_bias_scale,
        dropout,
        rnn_hidden_size,
    ):
        """
        Prepare the trainable parameters of the Prediction Network.

        Args:
            vocab_size: Vocab size (excluding the blank token).
            pred_n_hidden: Hidden size of the RNNs.
            pred_rnn_layers: Number of RNN layers.
            forget_gate_bias: Whether to perform unit forget gate bias.
            t_max: Whether to perform Chrono LSTM init.
            norm: Type of normalization to perform in RNN.
            weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
                sometimes helps reduce variance between runs.
            hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                the default behaviour.
            dropout: Whether to apply dropout to RNN.
            rnn_hidden_size: the hidden size of the RNN, if not specified, pred_n_hidden would be used
        """
        if self.blank_as_pad:
            embed = torch.nn.Embedding(
                vocab_size + 1, pred_n_hidden, padding_idx=self.blank_idx
            )
        else:
            embed = torch.nn.Embedding(vocab_size, pred_n_hidden)

        layers = torch.nn.ModuleDict(
            {
                "embed": embed,
                "dec_rnn": rnn.rnn(
                    input_size=pred_n_hidden,
                    hidden_size=(
                        rnn_hidden_size if rnn_hidden_size > 0 else pred_n_hidden
                    ),
                    num_layers=pred_rnn_layers,
                    norm=norm,
                    forget_gate_bias=forget_gate_bias,
                    t_max=t_max,
                    dropout=dropout,
                    weights_init_scale=weights_init_scale,
                    hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                    proj_size=pred_n_hidden if pred_n_hidden < rnn_hidden_size else 0,
                ),
            }
        )
        return layers

    def initialize_state(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the state of the LSTM layers, with same dtype and device as input `y`.
        LSTM accepts a tuple of 2 tensors as a state.

        Args:
            y: A torch.Tensor whose device the generated states will be placed on.

        Returns:
            Tuple of 2 tensors, each of shape [L, B, H], where

                L = Number of RNN layers

                B = Batch size

                H = Hidden size of RNN.
        """
        batch = y.size(0)
        if self.random_state_sampling and self.training:
            state = (
                torch.randn(
                    self.pred_rnn_layers,
                    batch,
                    self.pred_hidden,
                    dtype=y.dtype,
                    device=y.device,
                ),
                torch.randn(
                    self.pred_rnn_layers,
                    batch,
                    self.pred_hidden,
                    dtype=y.dtype,
                    device=y.device,
                ),
            )

        else:
            state = (
                torch.zeros(
                    self.pred_rnn_layers,
                    batch,
                    self.pred_hidden,
                    dtype=y.dtype,
                    device=y.device,
                ),
                torch.zeros(
                    self.pred_rnn_layers,
                    batch,
                    self.pred_hidden,
                    dtype=y.dtype,
                    device=y.device,
                ),
            )
        return state

    def score_hypothesis(
        self, hypothesis: rnnt_utils.Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Similar to the predict() method, instead this method scores a Hypothesis during beam search.
        Hypothesis is a dataclass representing one hypothesis in a Beam Search.

        Args:
            hypothesis: Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.

        Returns:
            Returns a tuple (y, states, lm_token) such that:
            y is a torch.Tensor of shape [1, 1, H] representing the score of the last token in the Hypothesis.
            state is a list of RNN states, each of shape [L, 1, H].
            lm_token is the final integer token of the hypothesis.
        """
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0].device
        else:
            _p = next(self.parameters())
            device = _p.device

        # parse "blank" tokens in hypothesis
        if (
            len(hypothesis.y_sequence) > 0
            and hypothesis.y_sequence[-1] == self.blank_idx
        ):
            blank_state = True
        else:
            blank_state = False

        # Convert last token of hypothesis to torch.Tensor
        target = torch.full(
            [1, 1],
            fill_value=hypothesis.y_sequence[-1],
            device=device,
            dtype=torch.long,
        )
        lm_token = target[:, -1]  # [1]

        # Convert current hypothesis into a tuple to preserve in cache
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            # Obtain score for target token and new states
            if blank_state:
                y, new_state = self.predict(
                    None, state=None, add_sos=False, batch_size=1
                )  # [1, 1, H]

            else:
                y, new_state = self.predict(
                    target, state=hypothesis.dec_state, add_sos=False, batch_size=1
                )  # [1, 1, H]

            y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    def batch_score_hypothesis(
        self,
        hypotheses: List[rnnt_utils.Hypothesis],
        cache: Dict[Tuple[int], Any],
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Used for batched beam search algorithms. Similar to score_hypothesis method.

        Args:
            hypothesis: List of Hypotheses. Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.

        Returns:
            Returns a tuple (batch_dec_out, batch_dec_states) such that:
                batch_dec_out: a list of torch.Tensor [1, H] representing the prediction network outputs for the last tokens in the Hypotheses.
                batch_dec_states: a list of list of RNN states, each of shape [L, B, H]. Represented as B x List[states].
        """
        final_batch = len(hypotheses)

        if final_batch == 0:
            raise ValueError("No hypotheses was provided for the batch!")

        _p = next(self.parameters())
        device = _p.device

        tokens = []
        to_process = []
        final = [None for _ in range(final_batch)]

        # For each hypothesis, cache the last token of the sequence and the current states
        for final_idx, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                final[final_idx] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                to_process.append((sequence, hyp.dec_state))

        if to_process:
            batch = len(to_process)

            # convert list of tokens to torch.Tensor, then reshape.
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(
                batch, -1
            )
            dec_states = self.batch_initialize_states(
                [d_state for _, d_state in to_process]
            )

            dec_out, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, 1, H], B x List([L, 1, H])

            # Update final states and cache shared by entire batch.
            processed_idx = 0
            for final_idx in range(final_batch):
                if final[final_idx] is None:
                    # Select sample's state from the batch state list
                    new_state = self.batch_select_state(dec_states, processed_idx)

                    # Cache [1, H] scores of the current y_j, and its corresponding state
                    final[final_idx] = (dec_out[processed_idx], new_state)
                    cache[to_process[processed_idx][0]] = (
                        dec_out[processed_idx],
                        new_state,
                    )

                    processed_idx += 1

        return [dec_out for dec_out, _ in final], [
            dec_states for _, dec_states in final
        ]

    def batch_initialize_states(
        self, decoder_states: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Creates a stacked decoder states to be passed to prediction network

        Args:
            decoder_states (list of list of list of torch.Tensor): list of decoder states
                [B, C, L, H]
                    - B: Batch size.
                    - C: e.g., for LSTM, this is 2: hidden and cell states
                    - L: Number of layers in prediction RNN.
                    - H: Dimensionality of the hidden state.

        Returns:
            batch_states (list of torch.Tensor): batch of decoder states
                [C x torch.Tensor[L x B x H]
        """
        # stack decoder states into tensor of shape [B x layers x L x H]
        # permute to the target shape [layers x L x B x H]
        stacked_states = torch.stack(
            [torch.stack(decoder_state) for decoder_state in decoder_states]
        )
        permuted_states = stacked_states.permute(1, 2, 0, 3)

        return list(permuted_states.contiguous())

    def batch_select_state(
        self, batch_states: List[torch.Tensor], idx: int
    ) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                ([L x (B, H)], [L x (B, H)])

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, H)], [L x (1, H)])
        """
        if batch_states is not None:
            return [state[:, idx] for state in batch_states]

        return None

    @classmethod
    def batch_aggregate_states_beam(
        cls,
        src_states: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        beam_size: int,
        indices: torch.Tensor,
        dst_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates decoder states based on the given indices.
        Args:
            src_states (Tuple[torch.Tensor, torch.Tensor]): source states of
                shape `([L x (batch_size * beam_size, H)], [L x (batch_size * beam_size, H)])`
            batch_size (int): The size of the batch.
            beam_size (int): The size of the beam.
            indices (torch.Tensor): A tensor of shape `(batch_size, beam_size)` containing
                the indices in beam that map the source states to the destination states.
            dst_states (Optional[Tuple[torch.Tensor, torch.Tensor]]): If provided, the method
                updates these tensors in-place.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
        Note:
            - The `indices` tensor is expanded to match the shape of the source states
            during the gathering operation.
        """
        layers_num = src_states[0].shape[0]
        layers_dim = src_states[0].shape[-1]

        beam_shape = torch.Size((layers_num, batch_size, beam_size, layers_dim))
        flat_shape = torch.Size((layers_num, batch_size * beam_size, layers_dim))

        # Expand indices to match the source states' shape
        indices_expanded = indices[None, :, :, None].expand(beam_shape)

        if dst_states is not None:
            # Perform in-place gathering into dst_states
            torch.gather(
                src_states[0].view(beam_shape),
                dim=2,
                index=indices_expanded,
                out=dst_states[0].view(beam_shape),
            )
            torch.gather(
                src_states[1].view(beam_shape),
                dim=2,
                index=indices_expanded,
                out=dst_states[1].view(beam_shape),
            )
            return dst_states

        # Gather and reshape into the output format
        return (
            torch.gather(
                src_states[0].view(beam_shape), dim=2, index=indices_expanded
            ).view(flat_shape),
            torch.gather(
                src_states[1].view(beam_shape), dim=2, index=indices_expanded
            ).view(flat_shape),
        )

    def batch_concat_states(
        self, batch_states: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Concatenate a batch of decoder state to a packed state.

        Args:
            batch_states (list): batch of decoder states
                B x ([L x (H)], [L x (H)])

        Returns:
            (tuple): decoder states
                (L x B x H, L x B x H)
        """
        state_list = []

        for state_id in range(len(batch_states[0])):
            batch_list = []
            for sample_id in range(len(batch_states)):
                tensor = (
                    torch.stack(batch_states[sample_id][state_id])
                    if not isinstance(batch_states[sample_id][state_id], torch.Tensor)
                    else batch_states[sample_id][state_id]
                )  # [L, H]
                tensor = tensor.unsqueeze(0)  # [1, L, H]
                batch_list.append(tensor)

            state_tensor = torch.cat(batch_list, 0)  # [B, L, H]
            state_tensor = state_tensor.transpose(1, 0)  # [L, B, H]
            state_list.append(state_tensor)

        return state_list

    @classmethod
    def batch_replace_states_mask(
        cls,
        src_states: Tuple[torch.Tensor, torch.Tensor],
        dst_states: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        other_src_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Replaces states in `dst_states` with states from `src_states` based on the given `mask`.

        Args:
            mask (torch.Tensor): When True, selects values from `src_states`, otherwise `out` or `other_src_states`(if provided).
            src_states (Tuple[torch.Tensor, torch.Tensor]): Values selected at indices where `mask` is True.
            dst_states (Tuple[torch.Tensor, torch.Tensor])): The output states.
            other_src_states (Tuple[torch.Tensor, torch.Tensor], optional): Values selected at indices where `mask` is False.

        Note:
            This operation is performed without CPU-GPU synchronization by using `torch.where`.
        """
        # same as `dst_states[i][mask] = src_states[i][mask]`, but non-blocking
        # we need to cast, since LSTM is calculated in fp16 even if autocast to bfloat16 is enabled

        other = other_src_states if other_src_states is not None else dst_states
        dtype = dst_states[0].dtype
        torch.where(
            mask.unsqueeze(0).unsqueeze(-1),
            src_states[0].to(dtype),
            other[0].to(dtype),
            out=dst_states[0],
        )
        torch.where(
            mask.unsqueeze(0).unsqueeze(-1),
            src_states[1].to(dtype),
            other[1].to(dtype),
            out=dst_states[1],
        )

    @classmethod
    def batch_replace_states_all(
        cls,
        src_states: Tuple[torch.Tensor, torch.Tensor],
        dst_states: Tuple[torch.Tensor, torch.Tensor],
    ):
        """Replace states in dst_states with states from src_states"""
        dst_states[0].copy_(src_states[0])
        dst_states[1].copy_(src_states[1])

    def batch_split_states(
        self, batch_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> list[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Split states into a list of states.
        Useful for splitting the final state for converting results of the decoding algorithm to Hypothesis class.
        """
        return list(
            zip(batch_states[0].split(1, dim=1), batch_states[1].split(1, dim=1))
        )

    def batch_copy_states(
        self,
        old_states: List[torch.Tensor],
        new_states: List[torch.Tensor],
        ids: List[int],
        value: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """Copy states from new state to old state at certain indices.

        Args:
            old_states(list): packed decoder states
                (L x B x H, L x B x H)

            new_states: packed decoder states
                (L x B x H, L x B x H)

            ids (list): List of indices to copy states at.

            value (optional float): If a value should be copied instead of a state slice, a float should be provided

        Returns:
            batch of decoder states with partial copy at ids (or a specific value).
                (L x B x H, L x B x H)
        """
        for state_id in range(len(old_states)):
            if value is None:
                old_states[state_id][:, ids, :] = new_states[state_id][:, ids, :]
            else:
                old_states[state_id][:, ids, :] *= 0.0
                old_states[state_id][:, ids, :] += value

        return old_states

    def mask_select_states(
        self, states: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return states by mask selection
        Args:
            states: states for the batch
            mask: boolean mask for selecting states; batch dimension should be the same as for states

        Returns:
            states filtered by mask
        """
        # LSTM in PyTorch returns a tuple of 2 tensors as a state
        return states[0][:, mask], states[1][:, mask]


class RNNTJoint(torch.nn.Module):
    """A Recurrent Neural Network Transducer Joint Network (RNN-T Joint Network).
    An RNN-T Joint network, comprised of a feedforward model.

    Args:
        jointnet: A dict-like object which contains the following key-value pairs.
            encoder_hidden: int specifying the hidden dimension of the encoder net.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            joint_hidden: int specifying the hidden dimension of the joint net
            activation: Activation function used in the joint step. Can be one of
            ['relu', 'tanh', 'sigmoid'].

            Optionally, it may also contain the following:
            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the joint net.

        num_classes: int, specifying the vocabulary size that the joint network must predict,
            excluding the RNNT blank token.

        vocabulary: Optional list of strings/tokens that comprise the vocabulary of the joint network.
            Unused and kept only for easy access for character based encoding RNNT models.

        log_softmax: Optional bool, set to None by default. If set as None, will compute the log_softmax()
            based on the value provided.

        preserve_memory: Optional bool, set to False by default. If the model crashes due to the memory
            intensive joint step, one might try this flag to empty the tensor cache in pytorch.

            Warning: This will make the forward-backward pass much slower than normal.
            It also might not fix the OOM if the GPU simply does not have enough memory to compute the joint.

        fuse_loss_wer: Optional bool, set to False by default.

            Fuses the joint forward, loss forward and
            wer forward steps. In doing so, it trades of speed for memory conservation by creating sub-batches
            of the provided batch of inputs, and performs Joint forward, loss forward and wer forward (optional),
            all on sub-batches, then collates results to be exactly equal to results from the entire batch.

            When this flag is set, prior to calling forward, the fields `loss` and `wer` (either one) *must*
            be set using the `RNNTJoint.set_loss()` or `RNNTJoint.set_wer()` methods.

            Further, when this flag is set, the following argument `fused_batch_size` *must* be provided
            as a non negative integer. This value refers to the size of the sub-batch.

            When the flag is set, the input and output signature of `forward()` of this method changes.
            Input - in addition to `encoder_outputs` (mandatory argument), the following arguments can be provided.

                - decoder_outputs (optional). Required if loss computation is required.

                - encoder_lengths (required)

                - transcripts (optional). Required for wer calculation.

                - transcript_lengths (optional). Required for wer calculation.

                - compute_wer (bool, default false). Whether to compute WER or not for the fused batch.

            Output - instead of the usual `joint` log prob tensor, the following results can be returned.

                - loss (optional). Returned if decoder_outputs, transcripts and transript_lengths are not None.

                - wer_numerator + wer_denominator (optional). Returned if transcripts, transcripts_lengths are provided
                    and compute_wer is set.

        fused_batch_size: Optional int, required if `fuse_loss_wer` flag is set. Determines the size of the
            sub-batches. Should be any value below the actual batch size per GPU.
        masking_prob: Optional float, indicating the probability of masking out decoder output in HAINAN
            (Hybrid Autoregressive Inference Transducer) model, described in https://arxiv.org/pdf/2410.02597
            Default to -1.0, which runs standard Joint network computation; if > 0, then masking out decoder output
            with the specified probability.
    """

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        num_extra_outputs: int = 0,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
        fuse_loss_wer: bool = False,
        fused_batch_size: Optional[int] = None,
        experimental_fuse_loss_wer: Any = None,
        masking_prob: float = -1.0,
    ):
        super().__init__()
        self.vocabulary = vocabulary

        self._vocab_size = num_classes
        self._num_extra_outputs = num_extra_outputs
        self._num_classes = num_classes +1+ num_extra_outputs

        self.masking_prob = masking_prob
        if self.masking_prob > 0.0:
            assert self.masking_prob < 1.0, "masking_prob must be between 0 and 1"

        if experimental_fuse_loss_wer is not None:
            # Override fuse_loss_wer from deprecated argument
            fuse_loss_wer = experimental_fuse_loss_wer

        self._fuse_loss_wer = fuse_loss_wer
        self._fused_batch_size = fused_batch_size

        if fuse_loss_wer and (fused_batch_size is None):
            raise ValueError(
                "If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!"
            )

        self._loss = None

        # Log softmax should be applied explicitly only for CPU
        self.log_softmax = log_softmax
        self.preserve_memory = preserve_memory

        # Required arguments
        self.encoder_hidden = jointnet["encoder_hidden"]
        self.pred_hidden = jointnet["pred_hidden"]
        self.joint_hidden = jointnet["joint_hidden"]
        self.activation = jointnet["activation"]

        # Optional arguments
        dropout = jointnet.get("dropout", 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net_modules(
            num_classes=self._num_classes,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.encoder_hidden,
            joint_n_hidden=self.joint_hidden,
            activation=self.activation,
            dropout=dropout,
        )

        # Flag needed for RNNT export support
        self._rnnt_export = False

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[torch.Tensor] = None,
        transcripts: Optional[torch.Tensor] = None,
        transcript_lengths: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # encoder = (B, T, D)
        # decoder = (B, D, U) if passed, else None
        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        if not self._fuse_loss_wer:
            if decoder_outputs is None:
                raise ValueError(
                    "decoder_outputs passed is None, and `fuse_loss_wer` is not set. "
                    "decoder_outputs can only be None for fused step!"
                )

            out = self.joint(encoder_outputs, decoder_outputs)  # [B, T, U, V + 1]
            return out

        else:
            # At least the loss module must be supplied during fused joint

            # When using fused joint step, both encoder and transcript lengths must be provided
            if (encoder_lengths is None) or (transcript_lengths is None):
                raise ValueError(
                    "`fuse_loss_wer` is set, therefore encoder and target lengths "
                    "must be provided as well!"
                )

            losses = []
            wers, wer_nums, wer_denoms = [], [], []
            target_lengths = []
            batch_size = int(encoder_outputs.size(0))  # actual batch size

            # Iterate over batch using fused_batch_size steps
            for batch_idx in range(0, batch_size, self._fused_batch_size):
                begin = batch_idx
                end = min(begin + self._fused_batch_size, batch_size)

                # Extract the sub batch inputs
                # sub_enc = encoder_outputs[begin:end, ...]
                # sub_transcripts = transcripts[begin:end, ...]
                sub_enc = encoder_outputs.narrow(
                    dim=0, start=begin, length=int(end - begin)
                )
                sub_transcripts = transcripts.narrow(
                    dim=0, start=begin, length=int(end - begin)
                )

                sub_enc_lens = encoder_lengths[begin:end]
                sub_transcript_lens = transcript_lengths[begin:end]

                # Sub transcripts does not need the full padding of the entire batch
                # Therefore reduce the decoder time steps to match
                max_sub_enc_length = sub_enc_lens.max()
                max_sub_transcript_length = sub_transcript_lens.max()

                if decoder_outputs is not None:
                    # Reduce encoder length to preserve computation
                    # Encoder: [sub-batch, T, D] -> [sub-batch, T', D]; T' < T
                    if sub_enc.shape[1] != max_sub_enc_length:
                        sub_enc = sub_enc.narrow(
                            dim=1, start=0, length=int(max_sub_enc_length)
                        )

                    # sub_dec = decoder_outputs[begin:end, ...]  # [sub-batch, U, D]
                    sub_dec = decoder_outputs.narrow(
                        dim=0, start=begin, length=int(end - begin)
                    )  # [sub-batch, U, D]

                    # Reduce decoder length to preserve computation
                    # Decoder: [sub-batch, U, D] -> [sub-batch, U', D]; U' < U
                    if sub_dec.shape[1] != max_sub_transcript_length + 1:
                        sub_dec = sub_dec.narrow(
                            dim=1, start=0, length=int(max_sub_transcript_length + 1)
                        )

                    # Perform joint => [sub-batch, T', U', V + 1]
                    sub_joint = self.joint(sub_enc, sub_dec)

                    del sub_dec

                    # Reduce transcript length to correct alignment
                    # Transcript: [sub-batch, L] -> [sub-batch, L']; L' <= L
                    if sub_transcripts.shape[1] != max_sub_transcript_length:
                        sub_transcripts = sub_transcripts.narrow(
                            dim=1, start=0, length=int(max_sub_transcript_length)
                        )

                    # Compute sub batch loss
                    # preserve loss reduction type

                    # override loss reduction to sum

                    # compute and preserve loss
                    loss_batch = self.loss(
                        log_probs=sub_joint,
                        targets=sub_transcripts,
                        input_lengths=sub_enc_lens,
                        target_lengths=sub_transcript_lens,
                    )

                    losses.append(loss_batch)
                    target_lengths.append(sub_transcript_lens)

                    # reset loss reduction type

                else:
                    losses = None

                # Update WER for sub batch

                del sub_enc, sub_transcripts, sub_enc_lens, sub_transcript_lens

            losses = self.loss.reduce(losses, target_lengths, batch_size)

            # Collect sub batch wer results

            return losses

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        """
        Project the Prediction Network (Decoder) output to the joint hidden dimension.

        Args:
            prednet_output: A torch.Tensor of shape [B, U, D]

        Returns:
            A torch.Tensor of shape [B, U, H]
        """
        raise NotImplementedError()

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the joint step of the network.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original paper.
            The original paper proposes the following steps :
            (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- *1
            *1 -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2) -> Sum [B, T, U, H] -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        
        return self.joint_after_projection(
            self.project_encoder(f), self.project_prednet(g)
        )

    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Project the encoder output to the joint hidden dimension.

        Args:
            encoder_output: A torch.Tensor of shape [B, T, D]

        Returns:
            A torch.Tensor of shape [B, T, H]
        """
        return self.enc(encoder_output)

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        """
        Project the Prediction Network (Decoder) output to the joint hidden dimension.

        Args:
            prednet_output: A torch.Tensor of shape [B, U, D]

        Returns:
            A torch.Tensor of shape [B, U, H]
        """
        return self.pred(prednet_output)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the joint step of the network after projection.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original paper.
            The original paper proposes the following steps :
            (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- \*1
            \*1 -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- \*1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- \*2
            (\*1, \*2) -> Sum [B, T, U, H] -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        g = g.unsqueeze(dim=1)  # (B, 1, U, H)

        if self.training and self.masking_prob > 0:
            [B, _, U, _] = g.shape
            rand = torch.rand([B, 1, U, 1]).to(g.device)
            rand = torch.gt(rand, self.masking_prob)
            g = g * rand

        inp = f + g  # [B, T, U, H]

        del f, g

        res = self.joint_net(inp)  # [B, T, U, V + 1]

        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()

        # If log_softmax is automatic
        if self.log_softmax is None:
            if not res.is_cuda:  # Use log softmax only if on CPU
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)
        else:
            if self.log_softmax:
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)
        return res

    def _joint_net_modules(
        self,
        num_classes,
        pred_n_hidden,
        enc_n_hidden,
        joint_n_hidden,
        activation,
        dropout,
    ):
        """
        Prepare the trainable modules of the Joint Network

        Args:
            num_classes: Number of output classes (vocab size) excluding the RNNT blank token.
            pred_n_hidden: Hidden size of the prediction network.
            enc_n_hidden: Hidden size of the encoder network.
            joint_n_hidden: Hidden size of the joint network.
            activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
            dropout: Dropout value to apply to joint.
        """
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)

        if activation not in ["relu", "sigmoid", "tanh"]:
            raise ValueError(
                "Unsupported activation for joint step - please pass one of "
                "[relu, sigmoid, tanh]"
            )

        activation = activation.lower()

        if activation == "relu":
            activation = torch.nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            activation = torch.nn.Sigmoid()
        elif activation == "tanh":
            activation = torch.nn.Tanh()

        layers = (
            [activation]
            + ([torch.nn.Dropout(p=dropout)] if dropout else [])
            + [torch.nn.Linear(joint_n_hidden, num_classes)]
        )
        return pred, enc, torch.nn.Sequential(*layers)

    @property
    def num_classes_with_blank(self):
        return self._num_classes

    @property
    def num_extra_outputs(self):
        return self._num_extra_outputs

    @property
    def loss(self):
        return self._loss

    def set_loss(self, loss):
        if not self._fuse_loss_wer:
            raise ValueError(
                "Attempting to set loss module even though `fuse_loss_wer` is not set!"
            )

        self._loss = loss

    @property
    def wer(self):
        return self._wer

    def set_wer(self, wer):
        if not self._fuse_loss_wer:
            raise ValueError(
                "Attempting to set WER module even though `fuse_loss_wer` is not set!"
            )

        self._wer = wer

    @property
    def fuse_loss_wer(self):
        return self._fuse_loss_wer

    def set_fuse_loss_wer(self, fuse_loss_wer, loss=None, metric=None):
        self._fuse_loss_wer = fuse_loss_wer

        self._loss = loss
        self._wer = metric

    @property
    def fused_batch_size(self):
        return self._fused_batch_size

    def set_fused_batch_size(self, fused_batch_size):
        self._fused_batch_size = fused_batch_size
