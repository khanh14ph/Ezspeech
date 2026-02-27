import torch
from torch.autograd import Function
from torch.nn import Module
from ezspeech.modules.losses.rnnt import gpu_rnnt

def certify_inputs(log_probs, labels, lengths, label_lengths):
    """Compact input validation."""
    assert log_probs.is_contiguous() and labels.is_contiguous(), "Inputs must be contiguous"
    assert log_probs.ndim == 4 and labels.ndim == 2, "Invalid input dimensions"
    assert lengths.shape[0] == log_probs.shape[0] == label_lengths.shape[0], "Batch size mismatch"
    
    max_T, max_U = lengths.max(), label_lengths.max()
    T, U = log_probs.shape[1:3]
    assert T == max_T, f"Input length mismatch: {T} vs {max_T}"
    assert U == max_U + 1, f"Output length mismatch: {U} vs {max_U + 1}"

class _TDTNumba(Function):
    @staticmethod
    def forward(ctx, label_acts, duration_acts, labels, act_lens, label_lens, blank, durations, reduction, fastemit_lambda, clamp, sigma, omega):
        certify_inputs(label_acts, labels, act_lens, label_lens)
        
        grads = torch.zeros_like(label_acts) if label_acts.requires_grad else None
        dur_grads = torch.zeros_like(duration_acts) if duration_acts.requires_grad else None
        costs = torch.zeros(label_acts.size(0), device=label_acts.device, dtype=label_acts.dtype)

        gpu_rnnt.tdt_loss_gpu(
            label_acts, duration_acts, labels, act_lens, label_lens, costs,
            grads, dur_grads, blank, durations, fastemit_lambda, clamp, sigma, omega, 0
        )

        if reduction in ['sum', 'mean']:
            costs = costs.sum().unsqueeze_(-1)
            if reduction == 'mean':
                costs /= label_acts.size(0)
                if grads is not None:
                    grads /= label_acts.size(0)
                    dur_grads /= label_acts.size(0)

        ctx.save_for_backward(grads, dur_grads)
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        label_grads, duration_grads = ctx.saved_tensors
        if grad_output is not None and label_grads is not None:
            grad_output = grad_output.view(-1, 1, 1, 1).to(label_grads)
            return label_grads.mul_(grad_output), duration_grads.mul_(grad_output), None, None, None, None, None, None, None, None, None, None

class TDTLossNumba(Module):
    def __init__(self, blank, durations=None, reduction='mean', fastemit_lambda=0.0, clamp=-1, sigma=0.0, omega=0.0):
        super().__init__()
        self.blank = blank
        self.durations = durations if durations is not None else []
        self.reduction = reduction
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp) if clamp > 0 else 0.0
        self.sigma = sigma
        self.omega = omega

    def forward(self, acts, labels, act_lens, label_lens):
        # Split acts into label and duration probabilities
        l_acts, d_acts = torch.split(acts, [acts.shape[-1] - len(self.durations), len(self.durations)], dim=-1)
        l_acts = l_acts.contiguous()
        # Duration acts require explicit log_softmax; label acts usually have it applied prior or within the joint
        d_acts = torch.nn.functional.log_softmax(d_acts, dim=-1).contiguous()

        return _TDTNumba.apply(
            l_acts, d_acts, labels, act_lens, label_lens,
            self.blank, self.durations, self.reduction,
            self.fastemit_lambda, self.clamp, self.sigma, self.omega
        )
