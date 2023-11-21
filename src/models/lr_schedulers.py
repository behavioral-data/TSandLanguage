
import math

import torch

class WarmupCosineDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, start_lr, end_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.cosine_decay_epochs = total_epochs - warmup_epochs
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.start_lr + (self.end_lr - self.start_lr) * (self.last_epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / self.cosine_decay_epochs
            lr = self.end_lr + 0.5 * (self.start_lr - self.end_lr) * (1 + math.cos(math.pi * progress))
        return [lr for _ in self.optimizer.param_groups]