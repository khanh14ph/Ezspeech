# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from ezspeech.modules.decoder.rnnt.rnnt_decoding.tdt_utils.ngram_lm import (
    NGramGPULanguageModel,
)
from ezspeech.modules.decoder.rnnt import rnnt_utils
from ezspeech.utils.cuda_python_utils import (
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
    run_nvrtc,
    with_conditional_node,
)
from ezspeech.modules.decoder.rnnt.rnnt_decoding.confidence_utils import (
    ConfidenceMethodMixin,
)

try:
    from cuda import cudart

    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False


class LoopLabelsState:
    """
    State for Loop Labels algorithm. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """

    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors

    all_durations: torch.Tensor

    encoder_output_projected: (
        torch.Tensor
    )  # projected output from the encoder for decoding algorithm
    encoder_output_length: (
        torch.Tensor
    )  # length of the (projected) output from the encoder

    labels: torch.Tensor  # storage for current labels
    scores: torch.Tensor  # storage for current scores

    batch_indices: (
        torch.Tensor
    )  # indices of elements in batch (constant, range [0, batch_size-1])

    time_indices: torch.Tensor  # current time indices for each element in batch
    safe_time_indices: (
        torch.Tensor
    )  # current time indices, but guaranteed to be < encoder_output_length
    time_indices_current_labels: (
        torch.Tensor
    )  # time indices for found labels (corresponding to `labels` field)
    last_timesteps: (
        torch.Tensor
    )  # indices of the last timesteps for each element (encoder_output_length - 1)

    active_mask: (
        torch.Tensor
    )  # mask for active hypotheses (the decoding is finished for the utterance if it is False)
    advance_mask: (
        torch.Tensor
    )  # mask for "advancing" hypotheses (blank is found for the element on the current step)
    blank_mask: torch.Tensor  # if the element is blank
    # if the element was active on the previous step: to identify the end of decoding and store final hidden state
    active_mask_prev: torch.Tensor
    became_inactive_mask: (
        torch.Tensor
    )  # mask for elements that became inactive (end of decoding)

    active_mask_any: (
        torch.Tensor
    )  # 0-dim bool tensor, condition for outer loop ('any element is still active')
    advance_mask_any: (
        torch.Tensor
    )  # 0-dim bool tensor, condition for inner loop ('should advance any index')

    last_decoder_state: Any  # last state from the decoder, needed for the output
    decoder_state: Any  # current decoder state
    decoder_output: torch.Tensor  # output from the decoder (projected)

    batched_hyps: rnnt_utils.BatchedHyps  # batched hypotheses - decoding result

    batch_lm_states: Optional[torch.Tensor] = None
    lm_scores: Optional[torch.Tensor] = None
    batch_lm_states_candidates: Optional[torch.Tensor] = None
    durations: Optional[torch.Tensor] = None  # storage for current predicted durations

    def __init__(
        self,
        batch_size: int,
        max_time: int,
        encoder_dim: int,
        max_symbols: int,
        device: torch.device,
        float_dtype: torch.dtype,
        logits_dim: int,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        include_duration_confidence: bool = False,
        include_duration: bool = False,
    ):
        """

        Args:
            batch_size: batch size for encoder output storage
            max_time: maximum time for encoder output storage
            encoder_dim: last dimension for encoder output storage (projected encoder output)
            max_symbols: max symbols per step (to avoid infinite looping and pre-allocate storage)
            device: device to store tensors
            float_dtype: default float dtype for tensors (should match projected encoder output)
            logits_dim: output dimension for Joint
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            include_duration_confidence: if duration confidence is needed to be added to the frame confidence
            include_duration: if predicted token durations are needed to be added to the Hypothesis object
        """
        self.device = device
        self.float_dtype = float_dtype
        self.batch_size = batch_size
        self.max_time = max_time

        self.encoder_output_projected = torch.zeros(
            (self.batch_size, self.max_time, encoder_dim),
            dtype=float_dtype,
            device=self.device,
        )
        self.encoder_output_length = torch.zeros(
            (self.batch_size,), dtype=torch.long, device=self.device
        )

        self.labels = torch.zeros(
            [self.batch_size], dtype=torch.long, device=self.device
        )
        self.scores = torch.zeros(
            [self.batch_size], dtype=float_dtype, device=self.device
        )

        # indices of elements in batch (constant)
        self.batch_indices = torch.arange(
            self.batch_size, dtype=torch.long, device=self.device
        )

        self.time_indices = torch.zeros_like(self.batch_indices)
        self.safe_time_indices = torch.zeros_like(self.batch_indices)
        self.time_indices_current_labels = torch.zeros_like(self.time_indices)
        self.last_timesteps = torch.zeros_like(self.time_indices)

        self.active_mask = torch.zeros(
            [self.batch_size], dtype=torch.bool, device=self.device
        )
        self.advance_mask = torch.zeros_like(self.active_mask)
        self.blank_mask = torch.zeros_like(self.active_mask)
        self.active_mask_prev = torch.zeros_like(self.active_mask)
        self.became_inactive_mask = torch.zeros_like(self.active_mask)

        self.active_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)
        self.advance_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)

        self.batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=self.batch_size,
            init_length=self.max_time * max_symbols,
            device=self.device,
            float_dtype=float_dtype,
        )
        if preserve_alignments or preserve_frame_confidence:
            self.alignments = rnnt_utils.BatchedAlignments(
                batch_size=batch_size,
                logits_dim=logits_dim,
                init_length=max_time * (max_symbols + 1),
                device=self.device,
                float_dtype=self.float_dtype,
                store_alignments=preserve_alignments,
                store_frame_confidence=preserve_frame_confidence,
                with_duration_confidence=include_duration_confidence,
            )
        else:
            self.alignments = None

        if include_duration:
            self.durations = torch.zeros(
                [self.batch_size], dtype=torch.long, device=self.device
            )
        else:
            self.durations = None

    def need_reinit(self, encoder_output_projected: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < encoder_output_projected.shape[0]
            or self.max_time < encoder_output_projected.shape[1]
            or self.device.index != encoder_output_projected.device.index
        )


@dataclass
class SeparateGraphsLoopLabels:
    """Class to store Cuda graphs for decoding when separate graphs are used"""

    before_outer_loop: torch.cuda.CUDAGraph = field(
        default_factory=torch.cuda.CUDAGraph
    )
    before_inner_loop: torch.cuda.CUDAGraph = field(
        default_factory=torch.cuda.CUDAGraph
    )
    inner_loop_code: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    after_inner_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)


class GreedyBatchedTDTLoopLabelsComputer(ConfidenceMethodMixin):
    """
    Label Looping algorithm implementation: optimized batched greedy decoding. Callable.
    Iterates over labels, on each step finding the next non-blank label
    (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
    to prediction network (with maximum possible batch size),
    which makes it especially useful for scaling the prediction network.
    During decoding all active hypotheses ("texts") have the same lengths.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs
    CUDA_PROGRAM_NAME = b"while_loop_labels_conditional_tdt.cu"

    class CudaGraphsMode:
        FULL_GRAPH = (
            "full_graph"  # Cuda graphs with conditional nodes, fastest implementation
        )
        NO_WHILE_LOOPS = (
            "no_while_loops"  # Decoding with PyTorch while loops + partial Cuda graphs
        )
        NO_GRAPHS = "no_graphs"  # decoding without graphs, stateful implementation, only for testing purposes

    separate_graphs: Optional[SeparateGraphsLoopLabels]
    full_graph: Optional[torch.cuda.CUDAGraph]
    cuda_graphs_mode: Optional[CudaGraphsMode]
    state: Optional[LoopLabelsState]
    ngram_lm_batch: Optional[NGramGPULanguageModel]

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        durations: Union[list[int], ListConfig[int]],
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        include_duration: bool = False,
        include_duration_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
        allow_cuda_graphs: bool = True,
        ngram_lm_model: Optional[str | Path] = None,
        ngram_lm_alpha: float = 0.0,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            durations: list of TDT durations, e.g., [0, 1, 2, 4, 8]
            max_symbols_per_step: max symbols to emit on each step (to avoid infinite looping)
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            include_duration: if predicted token durations are needed to be added to the Hypothesis object
            include_duration_confidence: if duration confidence is needed to be added to the frame confidence
            confidence_method_cfg: config for the confidence
            ngram_lm_model: optional n-gram language model (LM) file to use for decoding
            ngram_lm_alpha: LM weight
        """
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        # keep durations on CPU to avoid side effects in multi-gpu environment
        self.durations = torch.tensor(list(durations), device="cpu").to(torch.long)
        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self.allow_cuda_graphs = allow_cuda_graphs
        self.include_duration = include_duration
        self.include_duration_confidence = include_duration_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only

        self.state = None
        self.full_graph = None
        self.separate_graphs = None

        self.cuda_graphs_mode = None
        self.maybe_enable_cuda_graphs()

        if ngram_lm_model is not None:
            assert (
                self._blank_index
                == self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            )
            self.ngram_lm_batch = NGramGPULanguageModel.from_file(
                lm_path=ngram_lm_model, vocab_size=self._blank_index
            )
        else:
            self.ngram_lm_batch = None
        self.ngram_lm_alpha = ngram_lm_alpha

    def maybe_enable_cuda_graphs(self):
        """Enable CUDA graphs if conditions met"""
        if self.cuda_graphs_mode is not None:
            # CUDA graphs are already enabled
            return

        if not self.allow_cuda_graphs:
            self.cuda_graphs_mode = None
        else:
            # cuda graphs are allowed
            # check basic requirements for cuda graphs
            if self.max_symbols is None:
                print(
                    "Max symbols per step is None, which is not allowed with Cuda graphs. Setting to `10`"
                )
                self.max_symbols = 10
            # basic requirements met, need to check while loops
            try:
                check_cuda_python_cuda_graphs_conditional_nodes_supported()
                self.cuda_graphs_mode = self.CudaGraphsMode.FULL_GRAPH
            except (ImportError, ModuleNotFoundError, EnvironmentError) as e:
                print(
                    "No conditional node support for Cuda.\n"
                    "Cuda graphs with while loops are disabled, decoding speed will be slower\n"
                    f"Reason: {e}"
                )
                self.cuda_graphs_mode = self.CudaGraphsMode.NO_WHILE_LOOPS
        self.reset_cuda_graphs_state()

    def disable_cuda_graphs(self):
        """Disable CUDA graphs, can be used to disable graphs temporary, e.g., in training process"""
        if self.cuda_graphs_mode is None:
            # nothing to disable
            return
        self.cuda_graphs_mode = None
        self.reset_cuda_graphs_state()

    def reset_cuda_graphs_state(self):
        """Reset state to release memory (for CUDA graphs implementations)"""
        self.state = None
        self.full_graph = None
        self.separate_graphs = None

    def force_cuda_graphs_mode(self, mode: Optional[Union[str, CudaGraphsMode]]):
        """
        Method to set graphs mode. Use only for testing purposes.
        For debugging the algorithm use "no_graphs" mode, since it is impossible to debug CUDA graphs directly.
        """
        self.cuda_graphs_mode = self.CudaGraphsMode(mode) if mode is not None else None
        self.state = None

    def loop_labels_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        """
        Pure PyTorch implementation

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
        """
        batch_size, max_time, _unused = encoder_output.shape
        device = encoder_output.device
        if self.ngram_lm_batch is not None:
            self.ngram_lm_batch.to(
                device
            )  # ngram_lm_batch is nn.Module, but self is not; need to move manually

        # do not recalculate joint projection, project only once
        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype

        # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
        # init empty batched hypotheses
        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size,
            init_length=(
                max_time * self.max_symbols
                if self.max_symbols is not None
                else max_time
            ),
            device=device,
            float_dtype=float_dtype,
        )
        # sample state, will be replaced further when the decoding for hypothesis is done
        last_decoder_state = self.decoder.initialize_state(encoder_output_projected)
        # init alignments if necessary
        use_alignments = self.preserve_alignments or self.preserve_frame_confidence
        # always use alignments variable - for torch.jit adaptation, but keep it as minimal as possible
        alignments = rnnt_utils.BatchedAlignments(
            batch_size=batch_size,
            logits_dim=self.joint.num_classes_with_blank,
            init_length=(
                max_time * 2 if use_alignments else 1
            ),  # blank for each timestep + text tokens
            device=device,
            float_dtype=float_dtype,
            store_alignments=self.preserve_alignments,
            store_frame_confidence=self.preserve_frame_confidence,
            with_duration_confidence=self.include_duration_confidence,
        )

        # durations
        all_durations = self.durations.to(device, non_blocking=True)
        num_durations = all_durations.shape[0]

        # initial state, needed for torch.jit to compile (cannot handle None)
        state = self.decoder.initialize_state(encoder_output_projected)
        # indices of elements in batch (constant)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        # last found labels - initially <SOS> (<blank>) symbol
        labels = torch.full_like(batch_indices, fill_value=self._SOS)

        # time indices
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(
            time_indices
        )  # time indices, guaranteed to be < out_len
        time_indices_current_labels = torch.zeros_like(time_indices)
        last_timesteps = encoder_output_length - 1

        # masks for utterances in batch
        active_mask: torch.Tensor = encoder_output_length > 0
        advance_mask = torch.empty_like(active_mask)

        # for storing the last state we need to know what elements became "inactive" on this step
        active_mask_prev = torch.empty_like(active_mask)
        became_inactive_mask = torch.empty_like(active_mask)

        if self.ngram_lm_batch is not None:
            batch_lm_states = self.ngram_lm_batch.get_init_states(
                batch_size=batch_size, bos=True
            )

        # loop while there are active utterances
        while active_mask.any():
            active_mask_prev.copy_(active_mask, non_blocking=True)
            # stage 1: get decoder (prediction network) output
            decoder_output, state, *_ = self.decoder.predict(
                labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
            )
            decoder_output = self.joint.project_prednet(
                decoder_output
            )  # do not recalculate joint projection

            # stage 2: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            logits = (
                self.joint.joint_after_projection(
                    encoder_output_projected[
                        batch_indices, safe_time_indices
                    ].unsqueeze(1),
                    decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            scores, labels = logits[:, :-num_durations].max(dim=-1)
            if self.ngram_lm_batch is not None:
                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
                    states=batch_lm_states
                )  # vocab_size_no_blank
                lm_scores = lm_scores.to(dtype=float_dtype)
                # combined scores with LM - without blank
                scores_w_lm, labels_w_lm = (
                    logits[:, : -num_durations - 1] + self.ngram_lm_alpha * lm_scores
                ).max(dim=-1)
                # preserve "blank" / "non-blank" category
                torch.where(
                    labels == self._blank_index, labels, labels_w_lm, out=labels
                )
                torch.where(
                    labels == self._blank_index, scores, scores_w_lm, out=scores
                )

            jump_durations_indices = logits[:, -num_durations:].argmax(dim=-1)
            durations = all_durations[jump_durations_indices]

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            # for blank labels force duration >= 1
            durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
            time_indices_current_labels.copy_(time_indices, non_blocking=True)
            if use_alignments:
                alignments.add_results_masked_(
                    active_mask=active_mask,
                    time_indices=time_indices_current_labels,
                    logits=logits if self.preserve_alignments else None,
                    labels=labels if self.preserve_alignments else None,
                    confidence=(
                        torch.stack(
                            (
                                self._get_confidence_tensor(
                                    F.log_softmax(logits[:, :-num_durations], dim=-1)
                                ).to(dtype=float_dtype),
                                self._get_confidence_tensor(
                                    F.log_softmax(logits[:, -num_durations:], dim=-1)
                                ).to(dtype=float_dtype),
                            ),
                            dim=-1,
                        )
                        if self.include_duration_confidence
                        else (
                            self._get_confidence_tensor(
                                F.log_softmax(logits[:, :-num_durations], dim=-1)
                            ).to(dtype=float_dtype)
                            if self.preserve_frame_confidence
                            else None
                        )
                    ),
                )

            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            time_indices += durations
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            torch.less(time_indices, encoder_output_length, out=active_mask)
            torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # inner loop: find next non-blank labels (if exist)
            while advance_mask.any():
                # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
                # store current time indices to use further for storing the results
                torch.where(
                    advance_mask,
                    time_indices,
                    time_indices_current_labels,
                    out=time_indices_current_labels,
                )
                logits = (
                    self.joint.joint_after_projection(
                        encoder_output_projected[
                            batch_indices, safe_time_indices
                        ].unsqueeze(1),
                        decoder_output,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                more_scores, more_labels = logits[:, :-num_durations].max(dim=-1)
                if self.ngram_lm_batch is not None:
                    # combined scores with LM - without blank
                    more_scores_w_lm, more_labels_w_lm = (
                        logits[:, : -num_durations - 1]
                        + self.ngram_lm_alpha * lm_scores
                    ).max(dim=-1)
                    # preserve "blank" / "non-blank" category
                    torch.where(
                        more_labels == self._blank_index,
                        more_labels,
                        more_labels_w_lm,
                        out=more_labels,
                    )

                # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
                torch.where(advance_mask, more_labels, labels, out=labels)
                # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
                torch.where(advance_mask, more_scores, scores, out=scores)
                jump_durations_indices = logits[:, -num_durations:].argmax(dim=-1)
                durations = all_durations[jump_durations_indices]

                if use_alignments:
                    alignments.add_results_masked_(
                        active_mask=advance_mask,
                        time_indices=time_indices_current_labels,
                        logits=logits if self.preserve_alignments else None,
                        labels=more_labels if self.preserve_alignments else None,
                        confidence=(
                            torch.stack(
                                (
                                    self._get_confidence_tensor(
                                        F.log_softmax(
                                            logits[:, :-num_durations], dim=-1
                                        )
                                    ).to(dtype=float_dtype),
                                    self._get_confidence_tensor(
                                        F.log_softmax(
                                            logits[:, -num_durations:], dim=-1
                                        )
                                    ).to(dtype=float_dtype),
                                ),
                                dim=-1,
                            )
                            if self.include_duration_confidence
                            else (
                                self._get_confidence_tensor(
                                    F.log_softmax(logits[:, :-num_durations], dim=-1)
                                ).to(dtype=float_dtype)
                                if self.preserve_frame_confidence
                                else None
                            )
                        ),
                    )

                blank_mask = labels == self._blank_index
                # for blank labels force duration >= 1
                durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
                # same as time_indices[advance_mask] += durations[advance_mask], but non-blocking
                torch.where(
                    advance_mask,
                    time_indices + durations,
                    time_indices,
                    out=time_indices,
                )
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, encoder_output_length, out=active_mask)
                torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # stage 3: filter labels and state, store hypotheses
            # select states for hyps that became inactive (is it necessary?)
            # this seems to be redundant, but used in the `loop_frames` output
            torch.ne(active_mask, active_mask_prev, out=became_inactive_mask)
            self.decoder.batch_replace_states_mask(
                src_states=state,
                dst_states=last_decoder_state,
                mask=became_inactive_mask,
            )

            # store hypotheses
            if self.max_symbols is not None:
                # pre-allocated memory, no need for checks
                batched_hyps.add_results_masked_no_checks_(
                    active_mask,
                    labels,
                    time_indices_current_labels,
                    scores,
                    durations if self.include_duration else None,
                )
            else:
                # auto-adjusted storage
                batched_hyps.add_results_masked_(
                    active_mask,
                    labels,
                    time_indices_current_labels,
                    scores,
                    durations if self.include_duration else None,
                )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(
                            labels != self._blank_index,
                            batched_hyps.last_timestamp_lasts >= self.max_symbols,
                        ),
                        batched_hyps.last_timestamp == time_indices,
                    ),
                )
                time_indices += force_blank_mask  # emit blank => advance time indices
                # update safe_time_indices, non-blocking
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                # same as: active_mask = time_indices < encoder_output_length
                torch.less(time_indices, encoder_output_length, out=active_mask)
            if self.ngram_lm_batch is not None:
                # select necessary LM states based on chosen labels
                torch.where(
                    active_mask,
                    batch_lm_states_candidates[batch_indices, labels * active_mask],
                    batch_lm_states,
                    out=batch_lm_states,
                )
        if use_alignments:
            return batched_hyps, alignments, last_decoder_state
        return batched_hyps, None, last_decoder_state

    def loop_labels_cuda_graphs(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        """
        Implementation with CUDA graphs.

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
        """
        assert self.cuda_graphs_mode is not None
        # print("encoder_output",encoder_output.shape)
        # do not recalculate joint projection, project only once
        encoder_output = self.joint.project_encoder(encoder_output)
        current_batch_size = encoder_output.shape[0]
        current_max_time = encoder_output.shape[1]

        if torch.is_autocast_enabled():
            encoder_output = encoder_output.to(torch.get_autocast_gpu_dtype())

        # init or reinit graph
        if self.state is None or self.state.need_reinit(encoder_output):
            self._graph_reinitialize(encoder_output, encoder_output_length)

        # copy (projected) encoder output and lenghts
        self.state.encoder_output_projected[
            :current_batch_size, :current_max_time, ...
        ].copy_(encoder_output)
        self.state.encoder_output_length[: encoder_output_length.shape[0]].copy_(
            encoder_output_length
        )
        # set length to zero for elements outside the current batch
        self.state.encoder_output_length[current_batch_size:].fill_(0)
        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self.full_graph.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self.separate_graphs.before_outer_loop.replay()
            while self.state.active_mask_any.item():
                self.separate_graphs.before_inner_loop.replay()
                while self.state.advance_mask_any.item():
                    self.separate_graphs.inner_loop_code.replay()
                self.separate_graphs.after_inner_loop.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # this mode is only for testing purposes
            # manual loop instead of using graphs
            self._before_outer_loop()
            while self.state.active_mask_any.item():
                self._before_inner_loop_get_decoder_output()
                self._before_inner_loop_get_joint_output()
                while self.state.advance_mask_any.item():
                    self._inner_loop_code()
                self._after_inner_loop()
        else:
            raise NotImplementedError(f"Unknown graph mode: {self.cuda_graphs_mode}")

        return (
            self.state.batched_hyps,
            self.state.alignments,
            self.state.last_decoder_state,
        )

    @classmethod
    def _create_outer_while_loop_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the outer loop body (not all hypotheses are decoded).
        Condition: while(active_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
    
        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
    
        extern "C" __global__
        void outer_loop_labels_conditional(cudaGraphConditionalHandle handle, const bool *active_mask_any)
        {
         cudaGraphSetConditional(handle, *active_mask_any);
        }
        """
        return run_nvrtc(
            kernel_string, b"outer_loop_labels_conditional", cls.CUDA_PROGRAM_NAME
        )

    @classmethod
    def _create_inner_while_loop_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the inner loop body (not all non-blank labels found).
        Condition: while(advance_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
    
        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
    
        extern "C" __global__
        void inner_find_non_blank_conditional(cudaGraphConditionalHandle handle, const bool *advance_mask_any)
        {
         cudaGraphSetConditional(handle, *advance_mask_any);
        }
        """
        return run_nvrtc(
            kernel_string, b"inner_find_non_blank_conditional", cls.CUDA_PROGRAM_NAME
        )

    def _graph_reinitialize(
        self,
        encoder_output_projected: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ):
        batch_size, max_time, encoder_dim = encoder_output_projected.shape

        self.state = LoopLabelsState(
            batch_size=batch_size,
            max_time=max(max_time, self.INITIAL_MAX_TIME),
            encoder_dim=encoder_dim,
            max_symbols=self.max_symbols,
            device=encoder_output_projected.device,
            float_dtype=encoder_output_projected.dtype,
            logits_dim=self.joint.num_classes_with_blank,
            preserve_alignments=self.preserve_alignments,
            preserve_frame_confidence=self.preserve_frame_confidence,
            include_duration_confidence=self.include_duration_confidence,
            include_duration=self.include_duration,
        )
        self.state.all_durations = self.durations.to(self.state.device)

        self.state.last_decoder_state = self.decoder.initialize_state(
            encoder_output_projected
        )
        self.state.decoder_state = self.decoder.initialize_state(
            encoder_output_projected
        )
        decoder_output, *_ = self.decoder.predict(
            self.state.labels.unsqueeze(1),
            self.state.decoder_state,
            add_sos=False,
            batch_size=self.state.batch_size,
        )
        # to avoid recalculation of joint projection, store decoder output in state
        self.state.decoder_output = self.joint.project_prednet(decoder_output)

        if self.ngram_lm_batch is not None:
            device = encoder_output_projected.device
            float_dtype = encoder_output_projected.dtype
            vocab_size = self.ngram_lm_batch.vocab_size
            self.ngram_lm_batch.to(
                device
            )  # ngram_lm_batch is nn.Module, but self is not; need to move manually
            self.state.batch_lm_states = self.ngram_lm_batch.get_init_states(
                batch_size=self.state.batch_size, bos=True
            )
            self.state.batch_lm_states_candidates = torch.zeros(
                [batch_size, vocab_size], dtype=torch.long, device=device
            )
            self.state.lm_scores = torch.zeros(
                [batch_size, vocab_size], dtype=float_dtype, device=device
            )

        # warmup before graph compilation
        self._warmup_for_cuda_graphs()

        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self._full_graph_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self._partial_graphs_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # no graphs needed
            pass
        else:
            raise NotImplementedError

    def _warmup_for_cuda_graphs(self):
        """Warmup before compiling CUDA graphs"""
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        # 11 warmup steps required in DDP mode
        # see https://pytorch.org/docs/stable/notes/cuda.html#usage-with-distributeddataparallel
        num_runs = 11 if is_ddp else 3
        self.state.encoder_output_projected.fill_(0.0)
        self.state.encoder_output_length.fill_(1)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(num_runs):
                self._before_outer_loop()
                self._before_inner_loop_get_decoder_output()
                self._before_inner_loop_get_joint_output()
                self._inner_loop_code()
                self._after_inner_loop()
        torch.cuda.current_stream().wait_stream(s)
        self.state.encoder_output_length.fill_(0)

    def _partial_graphs_compile(self):
        """Compile decoding by parts"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.separate_graphs = SeparateGraphsLoopLabels()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.before_outer_loop,
                stream=stream_for_graph,
                capture_error_mode="thread_local",
            ),
        ):
            self._before_outer_loop()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.before_inner_loop,
                stream=stream_for_graph,
                capture_error_mode="thread_local",
            ),
        ):
            self._before_inner_loop_get_decoder_output()
            self._before_inner_loop_get_joint_output()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.inner_loop_code,
                stream=stream_for_graph,
                capture_error_mode="thread_local",
            ),
        ):
            self._inner_loop_code()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.after_inner_loop,
                stream=stream_for_graph,
                capture_error_mode="thread_local",
            ),
        ):
            self._after_inner_loop()

    def _full_graph_compile(self):
        """Compile full graph for decoding"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.full_graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.full_graph,
                stream=stream_for_graph,
                capture_error_mode="thread_local",
            ),
        ):
            self._before_outer_loop()

            capture_status, _, graph, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(
                    torch.cuda.current_stream(device=self.state.device).cuda_stream
                )
            )
            assert (
                capture_status
                == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
            )

            # capture: while self.active_mask_any:
            (outer_loop_conditional_handle,) = cu_call(
                cudart.cudaGraphConditionalHandleCreate(graph, 0, 0)
            )
            outer_loop_kernel = self._create_outer_while_loop_kernel()
            active_mask_any_ptr = np.array(
                [self.state.active_mask_any.data_ptr()], dtype=np.uint64
            )
            outer_loop_args = np.array(
                [
                    outer_loop_conditional_handle.getPtr(),
                    active_mask_any_ptr.ctypes.data,
                ],
                dtype=np.uint64,
            )

            # loop while there are active utterances
            # while self.active_mask_any:
            with with_conditional_node(
                outer_loop_kernel,
                outer_loop_args,
                outer_loop_conditional_handle,
                device=self.state.device,
            ):
                self._before_inner_loop_get_decoder_output()
                self._before_inner_loop_get_joint_output()
                # capture: while self.advance_mask_any.item():
                inner_while_loop_kernel = self._create_inner_while_loop_kernel()
                (inner_loop_conditional_handle,) = cu_call(
                    cudart.cudaGraphConditionalHandleCreate(graph, 0, 0)
                )
                advance_mask_any_ptr = np.array(
                    [self.state.advance_mask_any.data_ptr()], dtype=np.uint64
                )
                inner_loop_args = np.array(
                    [
                        inner_loop_conditional_handle.getPtr(),
                        advance_mask_any_ptr.ctypes.data,
                    ],
                    dtype=np.uint64,
                )
                # while self.advance_mask_any.item():
                with with_conditional_node(
                    inner_while_loop_kernel,
                    inner_loop_args,
                    inner_loop_conditional_handle,
                    device=self.state.device,
                ):
                    self._inner_loop_code()
                self._after_inner_loop()

    def _before_outer_loop(self):
        """Clear state and compute initial active mask"""
        self.state.batched_hyps.clear_()
        if self.state.alignments is not None:
            self.state.alignments.clear_()

        # initial state
        self.decoder.batch_replace_states_all(
            src_states=self.decoder.initialize_state(
                self.state.encoder_output_projected
            ),
            dst_states=self.state.decoder_state,
        )
        # initial state - lm
        if self.ngram_lm_batch is not None:
            self.state.batch_lm_states.copy_(
                self.ngram_lm_batch.get_init_states(
                    batch_size=self.state.batch_size, bos=True
                )
            )

        # last found labels - initially <SOS> (<blank>) symbol
        self.state.labels.fill_(self._SOS)
        self.state.scores.fill_(0.0)

        # time indices
        self.state.time_indices.fill_(0)
        self.state.safe_time_indices.fill_(
            0
        )  # safe time indices: guaranteed to be < encoder_output_length
        self.state.time_indices_current_labels.fill_(0)
        torch.sub(self.state.encoder_output_length, 1, out=self.state.last_timesteps)

        # masks for utterances in batch
        # same as: active_mask = self.encoder_output_length > 0
        torch.greater(self.state.encoder_output_length, 0, out=self.state.active_mask)

        # for storing the last state we need to know what elements became "inactive" on this step
        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def _before_inner_loop_get_decoder_output(self):
        """Get decoder output"""
        # stage 1: get decoder (prediction network) output
        decoder_output, new_state, *_ = self.decoder.predict(
            self.state.labels.unsqueeze(1),
            self.state.decoder_state,
            add_sos=False,
            batch_size=self.state.batch_size,
        )
        self.decoder.batch_replace_states_all(
            src_states=new_state, dst_states=self.state.decoder_state
        )
        decoder_output_projected = self.joint.project_prednet(
            decoder_output
        )  # do not recalculate joint projection
        self.state.decoder_output.copy_(decoder_output_projected)

        # get lm scores/states
        if self.ngram_lm_batch is not None:
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
                states=self.state.batch_lm_states
            )  # vocab_size_no_blank
            self.state.batch_lm_states_candidates.copy_(batch_lm_states_candidates)
            self.state.lm_scores.copy_(lm_scores.to(dtype=self.state.float_dtype))

    def _before_inner_loop_get_joint_output(self):
        """Get Joint output after decoder output, prepare inner loop to search for all next non-blank labels"""
        # stage 2: get joint output, iteratively seeking for non-blank labels
        # blank label in `labels` tensor means "end of hypothesis" (for this index)
        self.state.active_mask_prev.copy_(self.state.active_mask, non_blocking=True)
        logits = (
            self.joint.joint_after_projection(
                self.state.encoder_output_projected[
                    self.state.batch_indices, self.state.safe_time_indices
                ].unsqueeze(1),
                self.state.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # same as: scores, labels = logits[:, : -self.state.all_durations.shape[0]].max(-1)
        torch.max(
            logits[:, : -self.state.all_durations.shape[0]],
            dim=-1,
            out=(self.state.scores, self.state.labels),
        )
        if self.ngram_lm_batch is not None:
            scores_w_lm, labels_w_lm = (
                logits[:, : -self.state.all_durations.shape[0] - 1]
                + self.ngram_lm_alpha * self.state.lm_scores
            ).max(dim=-1)
            torch.where(
                self.state.labels == self._blank_index,
                self.state.labels,
                labels_w_lm,
                out=self.state.labels,
            )
            torch.where(
                self.state.labels == self._blank_index,
                self.state.scores,
                scores_w_lm,
                out=self.state.scores,
            )
        jump_durations_indices = logits[:, -self.state.all_durations.shape[0] :].argmax(
            dim=-1
        )
        durations = self.state.all_durations[jump_durations_indices]

        # search for non-blank labels using joint, advancing time indices for blank labels
        # checking max_symbols is not needed, since we already forced advancing time indices for such cases
        torch.eq(self.state.labels, self._blank_index, out=self.state.blank_mask)
        # blank_mask = self.labels == self._blank_index
        self.state.time_indices_current_labels.copy_(
            self.state.time_indices, non_blocking=True
        )
        # for blank labels force duration >= 1
        durations.masked_fill_(
            torch.logical_and(durations == 0, self.state.blank_mask), 1
        )

        if self.state.durations is not None:
            self.state.durations.copy_(durations, non_blocking=True)

        if self.state.alignments is not None:
            float_dtype = self.state.float_dtype
            self.state.alignments.add_results_masked_no_checks_(
                active_mask=self.state.active_mask,
                time_indices=self.state.time_indices_current_labels,
                logits=logits if self.preserve_alignments else None,
                labels=self.state.labels if self.preserve_alignments else None,
                confidence=(
                    torch.stack(
                        (
                            self._get_confidence_tensor(
                                F.log_softmax(
                                    logits[:, : -self.state.all_durations.shape[0]],
                                    dim=-1,
                                )
                            ).to(dtype=float_dtype),
                            self._get_confidence_tensor(
                                F.log_softmax(
                                    logits[:, -self.state.all_durations.shape[0] :],
                                    dim=-1,
                                )
                            ).to(dtype=float_dtype),
                        ),
                        dim=-1,
                    )
                    if self.include_duration_confidence
                    else (
                        self._get_confidence_tensor(
                            F.log_softmax(
                                logits[:, : -self.state.all_durations.shape[0]], dim=-1
                            )
                        ).to(dtype=float_dtype)
                        if self.preserve_frame_confidence
                        else None
                    )
                ),
            )

        # advance_mask is a mask for current batch for searching non-blank labels;
        # each element is True if non-blank symbol is not yet found AND we can increase the time index
        self.state.time_indices.add_(durations)
        torch.minimum(
            self.state.time_indices,
            self.state.last_timesteps,
            out=self.state.safe_time_indices,
        )
        torch.less(
            self.state.time_indices,
            self.state.encoder_output_length,
            out=self.state.active_mask,
        )
        torch.logical_and(
            self.state.active_mask, self.state.blank_mask, out=self.state.advance_mask
        )

        # inner loop: find next non-blank labels (if exist)
        # same as: self.advance_mask_any = advance_mask.any()
        torch.any(self.state.advance_mask, out=self.state.advance_mask_any)

    def _inner_loop_code(self):
        """Find next non-blank labels - one iteration"""
        # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
        # store current time indices to use further for storing the results
        torch.where(
            self.state.advance_mask,
            self.state.time_indices,
            self.state.time_indices_current_labels,
            out=self.state.time_indices_current_labels,
        )
        logits = (
            self.joint.joint_after_projection(
                self.state.encoder_output_projected[
                    self.state.batch_indices, self.state.safe_time_indices
                ].unsqueeze(1),
                self.state.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # get labels (greedy) and scores from current logits, replace labels/scores with new
        # labels[advance_mask] are blank, and we are looking for non-blank labels
        more_scores, more_labels = logits[:, : -self.state.all_durations.shape[0]].max(
            -1
        )
        if self.ngram_lm_batch is not None:
            # combined scores with LM - without blank
            more_scores_w_lm, more_labels_w_lm = (
                logits[:, : -self.state.all_durations.shape[0] - 1]
                + self.ngram_lm_alpha * self.state.lm_scores
            ).max(dim=-1)
            # preserve "blank" / "non-blank" category
            torch.where(
                more_labels == self._blank_index,
                more_labels,
                more_labels_w_lm,
                out=more_labels,
            )
            torch.where(
                more_labels == self._blank_index,
                more_scores,
                more_scores_w_lm,
                out=more_scores,
            )
        jump_durations_indices = logits[:, -self.state.all_durations.shape[0] :].argmax(
            dim=-1
        )
        durations = self.state.all_durations[jump_durations_indices]
        # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
        torch.where(
            self.state.advance_mask,
            more_labels,
            self.state.labels,
            out=self.state.labels,
        )
        # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
        torch.where(
            self.state.advance_mask,
            more_scores,
            self.state.scores,
            out=self.state.scores,
        )

        if self.state.alignments is not None:
            float_dtype = self.state.float_dtype
            self.state.alignments.add_results_masked_no_checks_(
                active_mask=self.state.advance_mask,
                time_indices=self.state.time_indices_current_labels,
                logits=logits if self.preserve_alignments else None,
                labels=more_labels if self.preserve_alignments else None,
                confidence=(
                    torch.stack(
                        (
                            self._get_confidence_tensor(
                                F.log_softmax(
                                    logits[:, : -self.state.all_durations.shape[0]],
                                    dim=-1,
                                )
                            ).to(dtype=float_dtype),
                            self._get_confidence_tensor(
                                F.log_softmax(
                                    logits[:, -self.state.all_durations.shape[0] :],
                                    dim=-1,
                                )
                            ).to(dtype=float_dtype),
                        ),
                        dim=-1,
                    )
                    if self.include_duration_confidence
                    else (
                        self._get_confidence_tensor(
                            F.log_softmax(
                                logits[:, : -self.state.all_durations.shape[0]], dim=-1
                            )
                        ).to(dtype=float_dtype)
                        if self.preserve_frame_confidence
                        else None
                    )
                ),
            )

        # blank_mask = self.labels == self._blank_index
        torch.eq(self.state.labels, self._blank_index, out=self.state.blank_mask)
        # for blank labels force duration >= 1
        durations.masked_fill_(
            torch.logical_and(durations == 0, self.state.blank_mask), 1
        )
        # self.time_indices += self.blank_mask
        torch.where(
            self.state.advance_mask,
            self.state.time_indices + durations,
            self.state.time_indices,
            out=self.state.time_indices,
        )

        if self.state.durations is not None:
            torch.where(
                self.state.advance_mask,
                durations,
                self.state.durations,
                out=self.state.durations,
            )

        torch.minimum(
            self.state.time_indices,
            self.state.last_timesteps,
            out=self.state.safe_time_indices,
        )
        torch.less(
            self.state.time_indices,
            self.state.encoder_output_length,
            out=self.state.active_mask,
        )
        torch.logical_and(
            self.state.active_mask, self.state.blank_mask, out=self.state.advance_mask
        )
        torch.any(self.state.advance_mask, out=self.state.advance_mask_any)

    def _after_inner_loop(self):
        """Store hypotheses, state for finished hypotheses, avoid looping"""
        # stage 3: filter labels and state, store hypotheses
        # select states for hyps that became inactive (is it necessary?)
        # this seems to be redundant, but used in the `loop_frames` output
        torch.ne(
            self.state.active_mask,
            self.state.active_mask_prev,
            out=self.state.became_inactive_mask,
        )
        self.decoder.batch_replace_states_mask(
            src_states=self.state.decoder_state,
            dst_states=self.state.last_decoder_state,
            mask=self.state.became_inactive_mask,
        )

        self.state.batched_hyps.add_results_masked_no_checks_(
            self.state.active_mask,
            self.state.labels,
            self.state.time_indices_current_labels,
            self.state.scores,
            self.state.durations,
        )

        if self.ngram_lm_batch is not None:
            # select necessary LM states based on chosen labels
            torch.where(
                self.state.active_mask,
                self.state.batch_lm_states_candidates[
                    self.state.batch_indices, self.state.labels * self.state.active_mask
                ],
                self.state.batch_lm_states,
                out=self.state.batch_lm_states,
            )

        # stage 4: to avoid looping, go to next frame after max_symbols emission
        # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
        # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
        force_blank_mask = torch.logical_and(
            self.state.active_mask,
            torch.logical_and(
                torch.logical_and(
                    self.state.labels != self._blank_index,
                    self.state.batched_hyps.last_timestamp_lasts >= self.max_symbols,
                ),
                self.state.batched_hyps.last_timestamp == self.state.time_indices,
            ),
        )
        self.state.time_indices.add_(
            force_blank_mask
        )  # emit blank => advance time indices
        # update safe_time_indices, non-blocking
        torch.minimum(
            self.state.time_indices,
            self.state.last_timesteps,
            out=self.state.safe_time_indices,
        )
        # same as: active_mask = time_indices < encoder_output_length
        torch.less(
            self.state.time_indices,
            self.state.encoder_output_length,
            out=self.state.active_mask,
        )
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        if self.cuda_graphs_mode is not None and x.device.type == "cuda":
            is_ddp = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
            # disable CUDA graphs if DDP and Mixed Precision are used
            ctx = (
                torch.amp.autocast(device_type="cuda", enabled=False)
                if is_ddp
                else nullcontext()
            )
            with ctx:
                # TODO(vbataev): fix issue with DDP+mixed precision, remove this restriction
                return self.loop_labels_cuda_graphs(
                    encoder_output=x, encoder_output_length=out_len
                )

        return self.loop_labels_torch(encoder_output=x, encoder_output_length=out_len)
