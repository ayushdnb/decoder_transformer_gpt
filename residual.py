# residual.py
# Residual wrapper: y = x + dropout(f(x_norm))
# Keeps block code clean and consistent.

# residual.py
import torch
import torch.nn as nn

class Residual(nn.Module):
    """Wrapper: y = x + dropout(f(x))"""
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, fn):
        # fn: callable mapping tensor -> tensor (caller handles pre-norm)
        return x + self.drop(fn(x))
