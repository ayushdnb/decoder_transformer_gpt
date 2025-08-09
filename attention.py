# Masked multi-head self-attention with RoPE(Q,K) and FlashAttention v2 (fallback to SDPA)

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func  # FA v2 interface (may vary by build)
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False


def apply_rope(q: torch.Tensor, k: torch.Tensor, rope_freqs: torch.Tensor):
    """
    q,k: [B, T, H, Dh]; Dh must be even
    rope_freqs: [T, 1, Dh] (precomputed cos/sin interleaved)
    Applies RoPE to (q,k).
    """
    cos, sin = rope_freqs[..., 0::2], rope_freqs[..., 1::2]

    def rope(x):
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        return torch.cat([x_even * cos - x_odd * sin,
                          x_even * sin + x_odd * cos], dim=-1)

    return rope(q), rope(k)


def build_rope_cache(T: int, Dh: int, theta: float, device, dtype):
    assert Dh % 2 == 0, "Head dim must be even for RoPE."
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)      # [T,1]
    idx = torch.arange(Dh // 2, device=device, dtype=dtype).unsqueeze(0)  # [1,Dh/2]
    freqs = 1.0 / (theta ** (idx * 2.0 / Dh))                            # [1, Dh/2]
    angles = pos * freqs                                                 # [T, Dh/2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rope = torch.stack([cos, sin], dim=-1).reshape(T, 1, Dh)
    return rope.to(dtype=dtype, device=device)


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        rope_theta: float,
        use_flash: bool,
        use_gqa: bool = False,  # unused in this minimal version
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flash = use_flash and HAS_FLASH
        self.rope_theta = rope_theta
        self.attn_drop = attn_dropout

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(attn_dropout)

        self.rope_cache: Optional[torch.Tensor] = None

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head)

    def _attend_flash(self, q, k, v):
        # Interface varies by build; this is illustrative. Fallback is robust.
        return flash_attn_func(q, k, v, causal=True, dropout_p=self.attn_drop if self.training else 0.0)

    def _attend_sdpa(self, q, k, v):
        B, T, H, Dh = q.shape
        q_ = q.permute(0, 2, 1, 3).reshape(B * H, T, Dh)
        k_ = k.permute(0, 2, 1, 3).reshape(B * H, T, Dh)
        v_ = v.permute(0, 2, 1, 3).reshape(B * H, T, Dh)
        out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.reshape(B, H, T, Dh).permute(0, 2, 1, 3)  # [B,T,H,Dh]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if (self.rope_cache is None) or (self.rope_cache.size(0) < T):
            self.rope_cache = build_rope_cache(T, self.d_head, self.rope_theta, device, dtype)

        q = self._shape_heads(self.wq(x))  # [B,T,H,Dh]
        k = self._shape_heads(self.wk(x))
        v = self._shape_heads(self.wv(x))

        # RoPE on Q,K
        rope = self.rope_cache[:T]         # [T,1,Dh]
        q, k = apply_rope(q, k, rope)

        if self.use_flash:
            try:
                out = self._attend_flash(q, k, v)  # [B,T,H,Dh]
            except Exception:
                self.use_flash = False
                out = self._attend_sdpa(q, k, v)
        else:
            out = self._attend_sdpa(q, k, v)

        out = out.reshape(B, T, D)
        out = self.wo(out)
        return self.drop(out)