# generate.py
# Fast sampler with top-p/top-k/temperature. Loads latest/best checkpoint.
import os, glob, argparse, torch
import torch.nn as nn
from model import TransformerLM, ModelArgs

BANNER = "[GEN] v2 sampler (_logits_only + tuple-safe) loaded"

def _logits_only(out):
    # Accepts: logits OR (logits, loss) OR [logits, loss]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

# -------- Tokenizer (SentencePiece) ----------
def load_tokenizer(spm_path: str):
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(spm_path)
        return sp.encode, sp.decode
    except Exception as e:
        print(f"[WARN] sentencepiece failed: {e}")
        return (
            lambda s: [int(x) for x in s.strip().split()],
            lambda ids: " ".join(map(str, ids))
        )

@torch.no_grad()
def sample(model: nn.Module, x: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
           top_k: int = 0, top_p: float = 1.0, eos_id: int = -1):
    print(BANNER)
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)

    for _ in range(max_new_tokens):
        out = model(x)
        logits = _logits_only(out)               # <- tuple-safe
        logits = logits[:, -1, :] / max(1e-8, temperature)
        probs = torch.softmax(logits, dim=-1)

        # top-k
        if top_k and top_k > 0:
            v, ix = torch.topk(probs, top_k)
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask.scatter_(1, ix, False)
            probs = probs.masked_fill(mask, 0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # top-p
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(filtered, num_samples=1)
            token = sorted_idx.gather(1, idx_next)
        else:
            token = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, token], dim=1)
        if eos_id >= 0 and token.item() == eos_id:
            break
    return x

def find_ckpt(path: str | None):
    if path and os.path.exists(path): return path
    best = sorted(glob.glob("train/runs/*/ckpt/best.pt"), key=os.path.getmtime, reverse=True)
    if best: return best[0]
    steps = sorted(glob.glob("train/runs/*/ckpt/step_*.pt"), key=os.path.getmtime, reverse=True)
    if steps: return steps[0]
    raise FileNotFoundError("No checkpoint found. Train first.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--spm", type=str, default="tokenizer.model")
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--eos_id", type=int, default=-1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = find_ckpt(args.ckpt)
    print(f"[LOAD] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    margs = ModelArgs(**ckpt["args"])
    model = TransformerLM(margs).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    encode, decode = load_tokenizer(args.spm)
    ids = encode(args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    out = sample(model, x, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.eos_id)
    gen_ids = out[0].tolist()
    text = decode(gen_ids)
    print("\n=== SAMPLE ===")
    print(text)

if __name__ == "__main__":
    main()
