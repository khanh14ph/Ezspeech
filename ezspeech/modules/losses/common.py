from torch.nn import CrossEntropyLoss
import torch.nn as nn
class CELoss(nn.Module):
    def __init__(self):
        self.ce_loss=CrossEntropyLoss()
    def forward(self,logits,labels):
        loss = self.ce_loss(logits.view(-1, self.config.vocab_size), labels.reshape(-1))
        return loss