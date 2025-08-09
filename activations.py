# activations.py
# SwiGLU activation: Linear(d -> 2d), then split: (a, b); act = SiLU(a) * b
# We implement just the gating and return reduced channel d (half of input).

# activations.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def forward(self, x):
        # x: [..., 2d]
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b  # returns [..., d]

