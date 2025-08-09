# =========================================
# Training configuration for StorytellerGPT
# RTX 3060 (6GB) friendly; excessive logging on.
# =========================================
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TrainConfig:
    # ---- Data ----
    train_bin: str = "data/train.bin"      # token ids (memmap)
    val_bin: str   = "data/val.bin"
    vocab_size: int = 50_000
    block_size: int = 512

    # ---- Optimization ----
    batch_size_tokens: int = 32 * 512      # effective tokens per step (micro * seq)
    micro_batch_size: int = 4              # per-step micro-batches (fits VRAM)
    lr_max: float = 3e-4                   # peak LR after warmup
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # ---- Scheduler (Hybrid) ----
    warmup_steps: int = 2000               # linear warmup
    total_steps: int = 500_000             # total optimizer steps target
    cosine_min_ratio: float = 0.1          # min LR at end of cosine
    plateau_patience: int = 1000           # steps without eval improvement to trigger plateau
    plateau_factor: float = 0.5            # multiply LR by this on plateau
    plateau_cooldown: int = 500            # steps to wait after a plateau hit
    min_lr: float = 3e-6                   # starting lr

    # ---- Runtime ----
    precision: str = "bf16"                # "bf16" | "fp16" | "fp32"
    compile: bool = False                  # torch.compile after 1 warm run
    seed: int = 1337
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    grad_accum_steps: Optional[int] = None # if None: computed from batch_size_tokens/(micro_batch*block)

    # ---- Logging / Eval / Save ----
    run_dir: str = "train/runs"
    print_every: int = 50                  # console prints this often (steps)
    eval_every: int = 500                  # run eval this often (steps)
    eval_tokens: int = 131072              # #tokens to eval loss on (fast estimate)
    save_every: int = 2000                 # checkpoint cadence
    keep_last: int = 5                     # how many checkpoints to keep