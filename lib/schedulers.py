import torch

torch.cuda.empty_cache()

# 自适应学习率调度器
class AdaptiveLearningRateScheduler:
    def __init__(self, optimizer, factor=0.1, patience=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = None
        self.num_bad_epochs = 0

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            if param_group['lr'] > new_lr:
                print(f'Reducing learning rate from {param_group["lr"]} to {new_lr}')
                param_group['lr'] = new_lr
        self.num_bad_epochs = 0
