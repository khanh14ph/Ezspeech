from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR,StepLR
class AdamOptimizer:
    def __init__(self,optimizer,scheduler):
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
    def get_optimizer(self,params):
        self.adam=AdamW(params=params,**self.optimizer_cfg)
        return self.adam
    def get_scheduler(self):
        return CosineAnnealingLR(optimizer=self.adam,**self.scheduler_cfg)