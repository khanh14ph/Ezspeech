from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
def init_weights(m, mode= 'xavier_uniform'):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:
                raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def complex_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    groups: int = 1,
) -> torch.Tensor:

    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors,
    # over only the first channel dimensions.
    # Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications,
    # where multiple sections of channels are multiplied independently
    # of one another (required for group convolutions).

    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.
    Useful for automatically computing the padding and stride for convolutions,
    where users may only provide an integer.
    Args:
        val: Value to cast into a tuple.
        n: Desired length of the tuple
    Returns:
        Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            error = f"Cannot cast tuple of length {len(out)} to length {n}."
            raise ValueError(error)
    else:
        return n * (val,)
def fft_convolution(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    padding: Union[int, Iterable[int]] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:

    """
    Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor
    after the convolution (in order ot mimic the PyTorch direct convolution).
    Args:
        signal: Input tensor to be convolved with the kernel.
        kernel: Convolution kernel.
        bias: Bias tensor to add to the output.
        padding: Number of zero samples to pad the
            input on the last dimension.
        stride: Stride size for computing output values.
    Returns:
        Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)

    # internal dilation offsets
    offset = torch.zeros(
        1,
        1,
        *dilation_,
        dtype=signal.dtype,
    )
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = F.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT,
    # we need the final dimension to have *even* length.
    # Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal_ = F.pad(signal, [0, 1])
    else:
        signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = F.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    dimension = tuple(range(2, signal.ndim))
    signal_fr = torch.fft.rfftn(signal_, dim=dimension)
    kernel_fr = torch.fft.rfftn(padded_kernel, dim=dimension)

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = torch.fft.irfftn(output_fr, dim=dimension)

    # Remove extra padded values
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    return output