# transformer.py
# One pre-norm decoder block: RMSNorm → Attn → Residual, RMSNorm → MLP(SwiGLU) → Residual

import torch
import torch.nn as nn

from attention import SelfAttention
from norm import RMSNorm
from activations import SwiGLU
from residual import Residual

class MLP(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.act = SwiGLU()  # applies gate inside
        self.fc2 = nn.Linear(hidden // 2, d_model, bias=False)  # SwiGLU halves channel post-gate
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # fc1 -> SwiGLU splits and gates -> reduced channels -> proj back
        x = self.fc1(x)
        x = self.act(x)          # [*, hidden//2]
        x = self.fc2(x)
        return self.drop(x)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_hidden: int,
        attn_dropout: float,
        resid_dropout: float,
        rope_theta: float,
        use_flash: bool,
        use_gqa: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = SelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            rope_theta=rope_theta,
            use_flash=use_flash,
            use_gqa=use_gqa,
        )
        self.resid1 = Residual(dropout=resid_dropout)

        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model=d_model, hidden=mlp_hidden, dropout=resid_dropout)
        self.resid2 = Residual(dropout=resid_dropout)

    def forward(self, x):
        # Pre-norm + residual
        x = self.resid1(x, lambda y: self.attn(self.norm1(y)))
        x = self.resid2(x, lambda y: self.mlp(self.norm2(y)))
        return x
