import torch
import torch.nn.functional as F
from torch.nn import Module

from ezspeech.modules.losses.rnnt.rnnt_pytorch import _TDTNumba


class TDTLoss(Module):
    """TDT loss accepting pre-split label and duration logits.

    This is the interface used by the hybrid CTC-TDT training model:
        loss = TDTLoss(...)
        loss(label_acts=..., duration_acts=..., targets=...,
             input_lengths=..., target_lengths=...)

    Args:
        blank_idx:        Index of the blank token (= vocab_size).
        durations:        List of supported frame-advance durations, e.g. [0,1,2,3,4].
        reduction:        ``'mean'``, ``'sum'``, or ``'none'``.
        fastemit_lambda:  FastEmit regularisation weight.
        clamp:            Gradient clamp threshold (disabled if <= 0).
        sigma:            Multi-blank logit-undernormalisation weight.
        omega:            Weight for the auxiliary regular RNN-T loss term.
    """

    def __init__(
        self,
        blank_idx: int,
        durations: list,
        reduction: str = 'mean',
        fastemit_lambda: float = 0.0,
        clamp: float = -1.0,
        sigma: float = 0.0,
        omega: float = 0.0,
    ):
        super().__init__()
        self.blank = blank_idx
        self.durations = durations
        self.reduction = reduction
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp)
        self.sigma = sigma
        self.omega = omega

    def forward(
        self,
        label_acts: torch.Tensor,       # (B, T, U+1, V+1)  raw logits
        duration_acts: torch.Tensor,    # (B, T, U+1, D)     raw logits
        targets: torch.Tensor,          # (B, U)
        input_lengths: torch.Tensor,    # (B,)
        target_lengths: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        label_acts    = label_acts.contiguous()
        duration_acts = F.log_softmax(duration_acts, dim=-1).contiguous()
        targets       = targets.int()
        input_lengths = input_lengths.int()
        target_lengths = target_lengths.int()

        return _TDTNumba.apply(
            label_acts, duration_acts, targets,
            input_lengths, target_lengths,
            self.blank, self.durations, self.reduction,
            self.fastemit_lambda, self.clamp, self.sigma, self.omega,
        )


from numba import cuda
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