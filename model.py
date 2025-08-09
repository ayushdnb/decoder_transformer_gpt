# Decoder-only LM: Embedding → [N x TransformerBlock] → Final RMSNorm → Tied LM Head
# Strictly pre-norm; RoPE(Q,K); FlashAttention w/ SDPA fallback.

from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TransformerBlock
from norm import RMSNorm

@dataclass
class ModelArgs:
    vocab_size: int = 50_000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    block_size: int = 512

    rope_theta: float = 10_000.0
    mlp_mult: float = 2.7     # 2.7 * d_model ~ LLaMA-ish
    dropout: float = 0.1
    attn_dropout: float = 0.0
    dtype: str = "bfloat16"   # "float16" if bf16 not available
    use_flash: bool = True
    use_gqa: bool = False     # future-proof; here q=k=v heads

    tie_weights: bool = True
    checkpointing: bool = True

class TransformerLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        d = args.d_model
        V = args.vocab_size

        # Embedding matrix (weight-tied with lm_head)
        self.tok_emb = nn.Embedding(V, d)
        self.drop = nn.Dropout(args.dropout)

        # Stack of decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d,
                n_heads=args.n_heads,
                mlp_hidden=int(math.ceil(args.mlp_mult * d / 64)) * 64,
                attn_dropout=args.attn_dropout,
                resid_dropout=args.dropout,
                rope_theta=args.rope_theta,
                use_flash=args.use_flash,
                use_gqa=args.use_gqa,
            )
            for _ in range(args.n_layers)
        ])

        self.final_norm = RMSNorm(d)
        # LM head shares weights with embedding by default
        self.lm_head = nn.Linear(d, V, bias=False)
        if args.tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        idx: [B, T] token ids (T <= block_size)
        targets: [B, T] (optional) for xent loss
        """
        B, T = idx.shape
        assert T <= self.args.block_size, "Sequence length exceeds block size."

        x = self.tok_emb(idx)                 # [B, T, d]
        x = self.drop(x)

        # Blocks (use gradient checkpointing to fit 6GB, if enabled)
        for block in self.blocks:
            if self.args.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.final_norm(x)                # [B, T, d]
        logits = self.lm_head(x)              # [B, T, V]

        if targets is None:
            return logits, None

        # Tripwires
        assert logits.dim() == 3 and logits.size(0) == B and logits.size(1) == T
        assert logits.size(-1) == self.args.vocab_size
        assert targets.dtype == torch.long

        # Cross-entropy over last dim
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        return logits, loss