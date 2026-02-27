import multiprocessing
import random
from typing import Optional, Tuple

import numba
import torch
from numba import cuda

from ezspeech.modules.losses.rnnt import global_constants,rnnt_helper
from ezspeech.modules.losses.rnnt import gpu_rnnt_kernel, reduce


class GPUTDT:
    def __init__(
        self,
        sigma: float,
        omega: float,
        num_durations: int,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace,
        tdt_workspace,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        stream,
    ):
        self.sigma = sigma
        self.omega = omega
        self.num_durations = num_durations
        self.minibatch_ = minibatch
        self.maxT_ = maxT
        self.maxU_ = maxU
        self.alphabet_size_ = alphabet_size
        self.blank_ = blank
        self.fastemit_lambda_ = fastemit_lambda
        self.clamp_ = abs(clamp)
        self.stream_ = stream

        # Memory mapping
        self.gpu_workspace = cuda.as_cuda_array(workspace)
        self.tdt_workspace = cuda.as_cuda_array(tdt_workspace)

        # Thread management
        _torch_num_threads = torch.get_num_threads()
        if num_threads > 0:
            numba.set_num_threads(min(multiprocessing.cpu_count(), num_threads))
        self.num_threads_ = numba.get_num_threads()
        torch.set_num_threads(_torch_num_threads)

    def _prepare_workspace(self) -> Tuple[int, Tuple[torch.Tensor, ...]]:
        used_offset = 0
        batch_area = self.maxT_ * self.maxU_ * self.minibatch_
        
        # Slice standard workspace
        denom = self.gpu_workspace[used_offset : used_offset + batch_area]
        used_offset += batch_area
        alphas = self.gpu_workspace[used_offset : used_offset + batch_area]
        used_offset += batch_area
        betas = self.gpu_workspace[used_offset : used_offset + batch_area]
        used_offset += batch_area
        llForward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_
        llBackward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_

        # Slice TDT workspace
        durations = self.tdt_workspace[: self.num_durations]

        return used_offset, (denom, alphas, betas, llForward, llBackward, durations)

    def log_softmax(self, acts: torch.Tensor, denom: torch.Tensor):
        rows = self.alphabet_size_
        cols = self.minibatch_ * self.maxT_ * self.maxU_
        reduce.reduce_max(acts, denom, rows=rows, cols=cols, minus=False, stream=self.stream_)
        reduce.reduce_exp(acts, denom, rows=rows, cols=cols, minus=True, stream=self.stream_)

    def compute_cost_and_score(
        self,
        label_acts: torch.Tensor,
        duration_acts: torch.Tensor,
        label_grads: Optional[torch.Tensor],
        duration_grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        
        training = label_grads is not None
        if training:
            label_grads *= 0.0
            if duration_grads is not None:
                duration_grads *= 0.0

        _, (denom, alphas, betas, llForward, llBackward, durations) = self._prepare_workspace()

        # 1. Log Softmax
        self.log_softmax(label_acts, denom)

        # 2. Forward Pass (Alpha)
        # Random sampling to switch between standard RNNT and TDT
        use_standard_rnnt = random.uniform(0, 1) < self.omega

        if use_standard_rnnt:
            gpu_rnnt_kernel.compute_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                label_acts, denom, alphas, llForward, input_lengths, label_lengths, labels,
                self.minibatch_, self.maxT_, self.maxU_, self.alphabet_size_, self.blank_
            )
        else:
            gpu_rnnt_kernel.compute_tdt_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                label_acts, duration_acts, denom, self.sigma, alphas, llForward, input_lengths, 
                label_lengths, labels, self.minibatch_, self.maxT_, self.maxU_, 
                self.alphabet_size_, self.blank_, durations, self.num_durations
            )

        # 3. Backward Pass (Beta + Grads)
        if training:
            grad_blocks = self.minibatch_ * self.maxT_ * self.maxU_
            grad_threads = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE

            if use_standard_rnnt:
                gpu_rnnt_kernel.compute_betas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                    label_acts, denom, betas, llBackward, input_lengths, label_lengths, labels,
                    self.minibatch_, self.maxT_, self.maxU_, self.alphabet_size_, self.blank_
                )
                gpu_rnnt_kernel.compute_grad_kernel[grad_blocks, grad_threads, self.stream_, 0](
                    label_grads, label_acts, denom, alphas, betas, llForward, input_lengths, 
                    label_lengths, labels, self.minibatch_, self.maxT_, self.maxU_, 
                    self.alphabet_size_, self.blank_, self.fastemit_lambda_, self.clamp_
                )
            else:
                gpu_rnnt_kernel.compute_tdt_betas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                    label_acts, duration_acts, denom, self.sigma, betas, llBackward, input_lengths, 
                    label_lengths, labels, self.minibatch_, self.maxT_, self.maxU_, 
                    self.alphabet_size_, self.blank_, durations, self.num_durations
                )
                gpu_rnnt_kernel.compute_tdt_grad_kernel[grad_blocks, grad_threads, self.stream_, 0](
                    label_grads, duration_grads, label_acts, duration_acts, denom, self.sigma, 
                    alphas, betas, llForward, input_lengths, label_lengths, labels, self.minibatch_, 
                    self.maxT_, self.maxU_, self.alphabet_size_, self.blank_, durations, 
                    self.num_durations, self.fastemit_lambda_, self.clamp_
                )

        # 4. Final Cost Calculation
        t_per_block = min(costs.shape[0], 32)
        b_per_grid = (costs.shape[0] + (t_per_block - 1)) // t_per_block
        rnnt_helper.compute_costs_data[b_per_grid, t_per_block, self.stream_, 0](
            llForward, costs, self.fastemit_lambda_
        )
        self.stream_.synchronize()

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def cost_and_grad(self, label_acts, duration_acts, label_grads, duration_grads, costs, pad_labels, label_lengths, input_lengths):
        if any(x is None for x in [label_acts, duration_acts, label_grads, duration_grads, costs, pad_labels, label_lengths, input_lengths]):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE
        return self.compute_cost_and_score(label_acts, duration_acts, label_grads, duration_grads, costs, pad_labels, label_lengths, input_lengths)

    def score_forward(self, label_acts, duration_acts, costs, pad_labels, label_lengths, input_lengths):
        if any(x is None for x in [label_acts, duration_acts, costs, pad_labels, label_lengths, input_lengths]):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE
        return self.compute_cost_and_score(label_acts, duration_acts, None, None, costs, pad_labels, label_lengths, input_lengths)



def tdt_loss_gpu(
    label_acts: torch.Tensor,
    duration_acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    label_grads: torch.Tensor,
    duration_grads: torch.Tensor,
    blank_label: int,
    durations: list,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
    sigma: float,
    omega: float,
):
    """
    Wrapper method for accessing GPU TDT loss (https://arxiv.org/abs/2304.06795).

    CUDA implementation ported from [HawkAaron/warp-transducer](https://github.com/HawkAaron/warp-transducer).

    Args:
        label_acts: Activation tensor of shape [B, T, U, V], where V includes the blank symbol.
        duration_acts: Activation tensor of shape [B, T, U, D], where D is the number of durations.
        labels: Ground truth labels of shape [B, U].
        input_lengths: Lengths of the acoustic sequence as a vector of ints [B].
        label_lengths: Lengths of the target sequence as a vector of ints [B].
        costs: Zero vector of length [B] in which costs will be set.
        label_grads: Zero tensor of shape [B, T, U, V] where the gradient to label_acts will be set.
        duration_grads: Zero tensor of shape [B, T, U, D] where the gradient to duration_acts will be set.
        blank_label: Index of the standard blank token in the vocabulary.
        durations: A list of supported durations for TDT. Must include 0 and 1.
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
        num_threads: Number of threads for OpenMP.
        sigma: logit-undernormalization weight used in the multi-blank model. Refer to
            the multi-blank paper https://arxiv.org/abs/2304.06795 for detailed explanations.
        omega: weight for regular RNN-T loss
    """
    minibatch_size = label_acts.shape[0]
    maxT = label_acts.shape[1]
    maxU = label_acts.shape[2]
    alphabet_size = label_acts.shape[3]

    if hasattr(cuda, 'external_stream'):
        stream = cuda.external_stream(torch.cuda.current_stream(label_acts.device).cuda_stream)
    else:
        stream = cuda.default_stream()

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = rnnt_helper.get_workspace_size(maxT, maxU, minibatch_size, gpu=True)

    if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError("Invalid parameter passed when calculating working space memory")

    # Select GPU index
    cuda.select_device(label_acts.device.index)
    gpu_workspace = torch.zeros(gpu_size, device=label_acts.device, dtype=label_acts.dtype, requires_grad=False)

    tdt_workspace = torch.zeros(len(durations), device=label_acts.device, dtype=torch.long, requires_grad=False)

    for i in range(0, len(durations)):
        tdt_workspace[i] = durations[i]

    ### VIEW TENSORS AS VECTORS FOR POINTER INDEXING ###
    label_acts, label_acts_shape = rnnt_helper.flatten_tensor(label_acts)
    duration_acts, duration_acts_shape = rnnt_helper.flatten_tensor(duration_acts)

    wrapper = GPUTDT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=gpu_workspace,
        tdt_workspace=tdt_workspace,
        num_durations=len(durations),
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        stream=stream,
        sigma=sigma,
        omega=omega,
    )

    if label_grads is None:
        status = wrapper.score_forward(
            label_acts=label_acts.data,
            duration_acts=duration_acts.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        ### FLATTEN GRAD TENSOR ###
        label_grads, label_grads_shape = rnnt_helper.flatten_tensor(label_grads)
        duration_grads, duration_grads_shape = rnnt_helper.flatten_tensor(duration_grads)

        status = wrapper.cost_and_grad(
            label_acts=label_acts.data,
            duration_acts=duration_acts.data,
            label_grads=label_grads.data,
            duration_grads=duration_grads.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del gpu_workspace, tdt_workspace, wrapper
    return True