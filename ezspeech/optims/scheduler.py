import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class NoamAnnealing(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int = 320,
        warmup_steps: int = 10000,
        min_lr: float = 1e-8,
        max_lr: float = 1e-2,
        last_epoch: int = -1,
    ):
        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.min_lr = min_lr
        self.max_lr = max_lr if max_lr else 0.05 * d_model ** (-0.5)
        self.norm = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        super(NoamAnnealing, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that "
                    "was lower than the minimum learning rate."
                )

        new_lrs = [
            self._noam_annealing(initial_lr=initial_lr, step=step)
            for initial_lr in self.base_lrs
        ]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        out_lr = (
            initial_lr
            * self.norm
            * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        )

        out_lr = min(out_lr, self.max_lr)
        out_lr = max(out_lr, self.min_lr)

        return out_lr


class WarmupCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        min_lr,
        max_lr,
        warmup_steps,
        fix_steps,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.gap_lr = max_lr - min_lr
        self.warmup_step = warmup_steps
        self.fix_step = fix_steps
        super(WarmupCosineScheduler, self).__init__(optimizer, -1)

    def get_lr(self):
        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that "
                    "was lower than the minimum learning rate."
                )

        new_lrs = [
            self._cosine_annealing(initial_lr=initial_lr, step=step)
            for initial_lr in self.base_lrs
        ]
        return new_lrs

    def _cosine_annealing(self, initial_lr, step):
        if step < self.warmup_step:
            return self.min_lr + self.gap_lr * (step / self.warmup_step)
        elif step >= self.warmup_step and step < self.fix_step:
            return self.min_lr + 0.5 * self.gap_lr * (
                1
                + math.cos(
                    math.pi
                    * (step - self.warmup_step)
                    / (self.fix_step - self.warmup_step)
                )
            )
        else:
            return self.min_lr
