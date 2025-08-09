# ==================================================================================
# Trainer for decoder-only GPT on token .bin memmaps.
# - Hybrid LR: Warmup → Cosine; ReduceLROnPlateau override on eval stagnation.
# - AMP (bf16/fp16), grad accumulation, grad clipping, checkpointing.
# - Loud console logs + CSV/TXT logs in train/runs/<timestamp>/.
# - Windows/RTX 3060 friendly; ASCII-only prints; deprecation-free AMP.
# ==================================================================================

import os, time, math, csv, json
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # ✅ new API
from torch.utils.data import Dataset, DataLoader

from train_config import TrainConfig
from model import TransformerLM, ModelArgs

# ---- Speed flags (TF32) ----
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ------------------------------
# Utilities
# ------------------------------

def seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_str():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ------------------------------
# Dataset: memmap token file
# ------------------------------

class BinDataset(Dataset):
    """
    Random-access rolling sequences of fixed length from a memmapped token file.
    Default dtype is uint16; change to np.int32 if your bin writer used int32.
    """
    def __init__(self, path: str, block_size: int, tokens_to_draw: Optional[int] = None, np_dtype=np.uint16):
        assert os.path.exists(path), f"Missing dataset: {path}"
        self.data = np.memmap(path, dtype=np_dtype, mode='r')
        self.block_size = block_size
        self.tokens_to_draw = tokens_to_draw or len(self.data)
        self.max_start = max(0, len(self.data) - block_size - 1)  # we need next-token labels
        if self.max_start <= 0:
            raise ValueError(f"{path}: not enough tokens for block_size={block_size}")

    def __len__(self):
        return max(1, self.tokens_to_draw // self.block_size)

    def __getitem__(self, idx):
        i = np.random.randint(0, self.max_start + 1)
        x = torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64))
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


# ------------------------------
# Hybrid LR Scheduler
# ------------------------------

class HybridScheduler:
    """Warmup → Cosine; ReduceLROnPlateau-like cut if eval stalls."""
    def __init__(self, optimizer, cfg: TrainConfig):
        self.opt = optimizer
        self.cfg = cfg
        self.step_num = 0
        self.best_eval = float('inf')
        self.last_improve = 0
        self.cooldown_until = -1

    def _lr_cosine(self, step):
        if step <= self.cfg.warmup_steps:
            # linear warmup from min_lr to lr_max
            warm_span = max(1, self.cfg.warmup_steps)
            return self.cfg.min_lr + (self.cfg.lr_max - self.cfg.min_lr) * (step / warm_span)
        progress = min(1.0, (step - self.cfg.warmup_steps) / max(1, self.cfg.total_steps - self.cfg.warmup_steps))
        min_lr = self.cfg.lr_max * self.cfg.cosine_min_ratio
        return min_lr + 0.5 * (self.cfg.lr_max - min_lr) * (1 + math.cos(math.pi * progress))

    def step(self):
        self.step_num += 1
        base_lr = self._lr_cosine(self.step_num)
        for pg in self.opt.param_groups:
            pg['lr'] = max(self.cfg.min_lr, base_lr)

    def notify_eval(self, eval_loss):
        if eval_loss + 1e-9 < self.best_eval:
            self.best_eval = eval_loss
            self.last_improve = self.step_num
            return
        if (self.step_num - self.last_improve) >= self.cfg.plateau_patience and (self.step_num >= self.cooldown_until):
            for pg in self.opt.param_groups:
                pg['lr'] = max(self.cfg.min_lr, pg['lr'] * self.cfg.plateau_factor)
            self.cooldown_until = self.step_num + self.cfg.plateau_cooldown


# ------------------------------
# Trainer
# ------------------------------

def main():
    cfg = TrainConfig()
    os.makedirs(cfg.run_dir, exist_ok=True)
    run_name = now_str()
    out_dir = os.path.join(cfg.run_dir, run_name)
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)

    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BOOT] Device: {device} | CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")

    # --- Data --- (change np_dtype if your bins are int32)
    train_ds = BinDataset(cfg.train_bin, cfg.block_size, np_dtype=np.uint16)
    val_ds   = BinDataset(cfg.val_bin,   cfg.block_size, tokens_to_draw=cfg.eval_tokens, np_dtype=np.uint16)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=4,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.micro_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # --- Model ---
    args = ModelArgs(
        vocab_size=cfg.vocab_size,
        d_model=512, n_layers=8, n_heads=8, block_size=cfg.block_size,
        rope_theta=10_000.0, mlp_mult=2.7, dropout=0.1, attn_dropout=0.0,
        dtype=cfg.precision, use_flash=True, use_gqa=False,
        tie_weights=True, checkpointing=True,
    )
    model = TransformerLM(args).to(device)

    if cfg.compile and hasattr(torch, "compile"):
        print("[INFO] torch.compile enabled (one warm run first).")
        model = torch.compile(model)

    # --- Optimizer (decoupled wd for matrices only) ---
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if (p.dim() >= 2 and "tok_emb" not in n) else no_decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.min_lr, betas=cfg.betas, fused=(device == "cuda"))  # start safe at min_lr

    sched = HybridScheduler(opt, cfg)

    # --- AMP ---
    use_fp16 = (cfg.precision == "fp16")
    use_bf16 = (cfg.precision == "bf16")
    scaler = GradScaler("cuda", enabled=use_fp16)  # fp16 only

    # --- Grad accumulation ---
    if cfg.grad_accum_steps is None:
        cfg.grad_accum_steps = max(1, cfg.batch_size_tokens // (cfg.micro_batch_size * cfg.block_size))
    print(f"[INFO] grad_accum_steps={cfg.grad_accum_steps} | micro_batch={cfg.micro_batch_size} | tokens/step≈{cfg.micro_batch_size*cfg.block_size*cfg.grad_accum_steps}")

    # --- Logging ---
    csv_path = os.path.join(out_dir, "train_log.csv")
    log_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["step","lr","train_loss","train_ppl","eval_loss","eval_ppl","tokens_sec","secs"])

    with open(os.path.join(out_dir, "run.txt"), "w") as f:
        f.write(f"Run: {run_name}\nDevice: {device}\nParams: {sum(p.numel() for p in model.parameters()):,}\n")

    # --- Train loop ---
    global_step = 0
    token_meter = deque(maxlen=50)
    t0 = time.time()
    best_eval = float('inf')

    def evaluate() -> float:
        model.eval()
        losses, tokens = [], 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=(torch.float16 if use_fp16 else torch.bfloat16), enabled=(use_fp16 or use_bf16)):
                    _, loss = model(x, y)
                losses.append(loss.item())
                tokens += x.numel()
                if tokens >= cfg.eval_tokens:
                    break
        model.train()
        return float(np.mean(losses)) if losses else float('inf')

    model.train()
    running_loss, running_steps = 0.0, 0

    while global_step < cfg.total_steps:
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=(torch.float16 if use_fp16 else torch.bfloat16), enabled=(use_fp16 or use_bf16)):
                logits, loss = model(x, y)
                # Tripwires (first few steps)
                if global_step < 3:
                    assert logits.dim() == 3, logits.shape
                    assert logits.size(0) == x.size(0) and logits.size(1) == x.size(1)
                    assert logits.size(2) == cfg.vocab_size
                    yv = y[y != -100]
                    if yv.numel() > 0:
                        print("[DBG] logits:", tuple(logits.shape), "y[min,max]=", int(yv.min().item()), int(yv.max().item()))

                true_loss = loss.item()
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += true_loss
            running_steps += 1

            # on optimizer step boundary
            if (global_step + 1) % cfg.grad_accum_steps == 0:
                # 1) update LR BEFORE stepping
                sched.step()

                # 2) unscale + clip
                if cfg.grad_clip is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                # 3) optimizer step
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                # log speed using the actual tokens processed this micro-batch * accum
                token_meter.append(x.size(0) * x.size(1) * cfg.grad_accum_steps)
                tok_per_sec = (sum(token_meter) / max(1e-6, (time.time() - t0)))

                steps_done = (global_step + 1) // cfg.grad_accum_steps
                if steps_done % cfg.print_every == 0:
                    lr = opt.param_groups[0]['lr']
                    avg_loss = running_loss / max(1, running_steps)
                    ppl = math.exp(min(20.0, avg_loss))
                    running_loss, running_steps = 0.0, 0
                    # Rough ETA based on tokens/sec and tokens/step
                    tokens_per_step = cfg.micro_batch_size * cfg.block_size * cfg.grad_accum_steps
                    steps_left = cfg.total_steps - steps_done
                    eta_sec = steps_left * (tokens_per_step / max(1e-6, tok_per_sec))
                    print(f"[STEP {steps_done:>7}] lr={lr:.3e} train_loss={avg_loss:.4f} ppl={ppl:.1f} tok/s={tok_per_sec:,.0f} ETA~{eta_sec/3600:.2f}h")
                    csv_writer.writerow([steps_done, lr, avg_loss, ppl, "", "", int(tok_per_sec), int(time.time()-t0)])
                    log_file.flush()

                # periodic eval
                if steps_done >= cfg.eval_every and (steps_done % cfg.eval_every == 0):
                    eval_loss = evaluate(); lr = opt.param_groups[0]['lr']
                    eval_ppl = math.exp(min(20.0, eval_loss)) if eval_loss < 20 else float('inf')
                    print(f"[EVAL @ {steps_done:>7}] eval_loss={eval_loss:.4f} ppl={eval_ppl:.1f} | lr={lr:.3e}")
                    csv_writer.writerow([steps_done, lr, "", "", eval_loss, eval_ppl, "", int(time.time()-t0)])
                    log_file.flush()

                    sched.notify_eval(eval_loss)
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        save_path = os.path.join(out_dir, "ckpt", "best.pt")
                        torch.save({
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "cfg": vars(cfg),
                            "args": vars(model.args),
                            "step": steps_done,
                        }, save_path)
                        print(f"[CKPT] New best saved: {save_path}")

                # periodic checkpoint
                if steps_done >= cfg.save_every and (steps_done % cfg.save_every == 0):
                    ckpt_path = os.path.join(out_dir, "ckpt", f"step_{steps_done}.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "cfg": vars(cfg),
                        "args": vars(model.args),
                        "step": steps_done,
                    }, ckpt_path)
                    print(f"[CKPT] Saved {ckpt_path}")

            global_step += 1
            if global_step >= cfg.total_steps:
                break

    log_file.close()
    print(f"[DONE] Run dir: {out_dir}")


if __name__ == "__main__":
    main()