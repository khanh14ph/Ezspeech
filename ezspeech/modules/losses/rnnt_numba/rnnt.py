# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#
# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing

import torch
from numba import cuda

from ezspeech.modules.losses.rnnt_numba.utils import global_constants, rnnt_helper
from ezspeech.modules.losses.rnnt_numba.utils.cpu_utils import cpu_rnnt
from ezspeech.modules.losses.rnnt_numba.utils.cuda_utils import gpu_rnnt

import torch
from torch.autograd import Function
from torch.nn import Module
def rnnt_loss_cpu(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    grads: torch.Tensor,
    blank_label: int,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
):
    """
    Wrapper method for accessing CPU RNNT loss.

    CPU implementation ported from [HawkAaron/warp-transducer](https://github.com/HawkAaron/warp-transducer).

    Args:
        acts: Activation tensor of shape [B, T, U, V+1].
        labels: Ground truth labels of shape [B, U].
        input_lengths: Lengths of the acoustic sequence as a vector of ints [B].
        label_lengths: Lengths of the target sequence as a vector of ints [B].
        costs: Zero vector of length [B] in which costs will be set.
        grads: Zero tensor of shape [B, T, U, V+1] where the gradient will be set.
        blank_label: Index of the blank token in the vocabulary.
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
        num_threads: Number of threads for OpenMP.
    """
    # aliases
    log_probs = acts
    flat_labels = labels

    minibatch_size = log_probs.shape[0]
    maxT = log_probs.shape[1]
    maxU = log_probs.shape[2]
    alphabet_size = log_probs.shape[3]

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = rnnt_helper.get_workspace_size(maxT, maxU, minibatch_size, gpu=False)
    if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError("Invalid parameter passed when calculating working space memory")

    cpu_workspace = torch.zeros(gpu_size, device=log_probs.device, dtype=log_probs.dtype, requires_grad=False)

    ### VIEW TENSORS AS VECTORS FOR POINTER INDEXING ###
    log_probs, acts_shape = rnnt_helper.flatten_tensor(log_probs)
    flat_labels, labels_shape = rnnt_helper.flatten_tensor(flat_labels)

    wrapper = cpu_rnnt.CPURNNT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=cpu_workspace,
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        batch_first=True,
    )

    if grads is None:
        status = wrapper.score_forward(
            log_probs=log_probs.data,
            costs=costs,
            flat_labels=flat_labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        ### FLATTEN GRAD TENSOR ###
        grads, grads_shape = rnnt_helper.flatten_tensor(grads)

        status = wrapper.cost_and_grad(
            log_probs=log_probs.data,
            grads=grads.data,
            costs=costs,
            flat_labels=flat_labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del cpu_workspace, wrapper
    return True


def rnnt_loss_gpu(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    grads: torch.Tensor,
    blank_label: int,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
):
    """
    Wrapper method for accessing GPU RNNT loss.

    CUDA implementation ported from [HawkAaron/warp-transducer](https://github.com/HawkAaron/warp-transducer).

    Args:
        acts: Activation tensor of shape [B, T, U, V+1].
        labels: Ground truth labels of shape [B, U].
        input_lengths: Lengths of the acoustic sequence as a vector of ints [B].
        label_lengths: Lengths of the target sequence as a vector of ints [B].
        costs: Zero vector of length [B] in which costs will be set.
        grads: Zero tensor of shape [B, T, U, V+1] where the gradient will be set.
        blank_label: Index of the blank token in the vocabulary.
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
        num_threads: Number of threads for OpenMP.
    """
    minibatch_size = acts.shape[0]
    maxT = acts.shape[1]
    maxU = acts.shape[2]
    alphabet_size = acts.shape[3]

    if hasattr(cuda, 'external_stream'):
        stream = cuda.external_stream(torch.cuda.current_stream(acts.device).cuda_stream)
    else:
        stream = cuda.default_stream()

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = rnnt_helper.get_workspace_size(maxT, maxU, minibatch_size, gpu=True)
    if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError("Invalid parameter passed when calculating working space memory")

    # Select GPU index
    cuda.select_device(acts.device.index)
    gpu_workspace = torch.zeros(gpu_size, device=acts.device, dtype=torch.float32, requires_grad=False)

    ### VIEW TENSORS AS VECTORS FOR POINTER INDEXING ###
    acts, acts_shape = rnnt_helper.flatten_tensor(acts)

    wrapper = gpu_rnnt.GPURNNT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=gpu_workspace,
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        stream=stream,
    )

    if grads is None:
        status = wrapper.score_forward(
            acts=acts.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        ### FLATTEN GRAD TENSOR ###
        grads, grads_shape = rnnt_helper.flatten_tensor(grads)

        status = wrapper.cost_and_grad(
            acts=acts.data,
            grads=grads.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del gpu_workspace, wrapper
    return True


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

    wrapper = gpu_rnnt.GPUTDT(
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

class _TDTNumba(Function):
    """
    Numba class for Token-and-Duration Transducer (TDT) loss (https://arxiv.org/abs/2304.06795)
    """

    @staticmethod
    def forward(
        ctx,
        label_acts,
        duration_acts,
        labels,
        act_lens,
        label_lens,
        blank,
        durations,
        fastemit_lambda,
        clamp,
        sigma,
        omega,
    ):
        """
        log_probs: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        durations: list of durations for TDT model, must include 0 and 1, e.g.
            [0, 1, 2, 3, 4].
        sigma: hyper-parameter for logit under-normalization method for training
            TDT models. Recommended value 0.05.
        omega: probability for sampling the standard RNN-T loss.
        Refer to https://arxiv.org/abs/2304.06795 for detailed explanations for
            the above parameters;
        """
        is_cuda = label_acts.is_cuda

        if clamp < 0:
            raise ValueError("`clamp` must be 0.0 or positive float value.")

        if is_cuda:
            loss_func = tdt_loss_gpu
        else:
            raise ValueError("TDT is not yet implemented for non CUDA computation.")

        label_grads = torch.zeros_like(label_acts) if label_acts.requires_grad else None
        duration_grads = torch.zeros_like(duration_acts) if duration_acts.requires_grad else None
        minibatch_size = label_acts.size(0)
        costs = torch.zeros(minibatch_size, device=label_acts.device, dtype=label_acts.dtype)

        loss_func(
            label_acts,
            duration_acts,
            labels=labels,
            input_lengths=act_lens,
            label_lengths=label_lens,
            costs=costs,
            label_grads=label_grads,
            duration_grads=duration_grads,
            blank_label=blank,
            durations=durations,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
            num_threads=0,
        )

        costs = costs.sum().unsqueeze_(-1)
        ctx.save_for_backward(label_grads, duration_grads)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        label_grads, duration_grads = ctx.saved_tensors
        if grad_output is not None and label_grads is not None:
            grad_output = grad_output.view(-1, 1, 1, 1).to(label_grads)
            return (
                label_grads.mul_(grad_output),
                duration_grads.mul_(grad_output),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

