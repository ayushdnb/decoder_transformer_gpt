# eval_gen_quality.py
# ------------------------------------------------------------
# Quick-and-dirty generation quality harness.
# - Loads your best/latest ckpt
# - Generates continuations for a set of prompts
# - Computes: conditional ppl (on the generated part), avg token entropy,
#   distinct-1/2/3, repetition stats (max trigram freq, repeat ratio),
#   length & end-of-sentence rate.
# - Writes results to CSV and prints a summary "quality gate".
# ------------------------------------------------------------
import os, glob, csv, math, argparse, re, statistics as stats
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import torch.nn as nn

from model import TransformerLM, ModelArgs

def _logits_only(out):
    return out[0] if isinstance(out, (tuple, list)) else out


# ---------- Tokenizer ----------
def load_tokenizer(spm_path: str):
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(spm_path)
        return sp.encode, sp.decode, sp.pad_id() if sp.pad_id() >= 0 else None
    except Exception as e:
        print(f"[WARN] sentencepiece not available or failed to load: {e}")
        # Fallback: whitespace tokenization (only for smoke tests)
        encode = lambda s: [int(x) for x in s.strip().split()]
        decode = lambda ids: " ".join(map(str, ids))
        return encode, decode, None


# ---------- Checkpoint finder ----------
def find_ckpt(path: str | None):
    if path and os.path.exists(path): return path
    best = sorted(glob.glob("train/runs/*/ckpt/best.pt"), key=os.path.getmtime, reverse=True)
    if best: return best[0]
    steps = sorted(glob.glob("train/runs/*/ckpt/step_*.pt"), key=os.path.getmtime, reverse=True)
    if steps: return steps[0]
    raise FileNotFoundError("No checkpoint found. Train first.")

# ---------- Metrics ----------
def distinct_n(ids, n):
    if len(ids) < n: return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return len(set(grams)) / max(1, len(grams))

def repetition_stats(ids):
    # trigram repetition profile + simple loopiness proxy
    if len(ids) < 3: return 0, 0.0
    grams = [tuple(ids[i:i+3]) for i in range(len(ids)-3+1)]
    cnt = Counter(grams)
    max_freq = max(cnt.values())
    repeat_ratio = 1.0 - (len(cnt) / max(1, len(grams)))
    return max_freq, repeat_ratio

def eos_rate(texts):
    return sum(t.strip().endswith(('.', '!', '?', '”', '"', "’")) for t in texts) / max(1, len(texts))

@torch.no_grad()
def conditional_ppl_and_entropy(model, ids, gen_len):
    """
    ids: tensor [1, T_total] = prompt + generated
    gen_len: number of generated tokens at end of sequence
    Computes NLL on the generated span only (teacher forcing).
    Returns: ppl, avg_token_entropy (bits)
    """
    device = next(model.parameters()).device
    ids = ids.to(device)
    T = ids.shape[1]
    assert gen_len > 0 and gen_len < T
    # logits for positions 0..T-2 predicting 1..T-1
    logits = _logits_only(model(ids[:, :-1])) # [1, T-1, V]
    targets = ids[:, 1:]         # [1, T-1]

    # mask to only the last gen_len positions
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, -gen_len:] = True

    logits = logits[mask]        # [G, V]
    targets = targets[mask]      # [G]
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [G]
    ppl = torch.exp(nll.mean()).item()

    # entropy in bits
    probs = torch.softmax(logits, dim=-1)
    entropy = (-probs * (probs.clamp_min(1e-12)).log()).sum(-1) / math.log(2.0)
    return ppl, entropy.mean().item()

# ---------- Sampler (greedy nucleus/top-k with per-step logits access) ----------
@torch.no_grad()
def generate_ids(model, start_ids, max_new_tokens=128, temperature=0.9, top_k=40, top_p=0.95, eos_id=-1):
    device = next(model.parameters()).device
    x = start_ids.to(device)
    for _ in range(max_new_tokens):
        logits = _logits_only(model(x))[:, -1, :] / max(1e-8, temperature)  
        probs = torch.softmax(logits, dim=-1)

        if top_k and top_k > 0:
            v, ix = torch.topk(probs, top_k)
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask.scatter_(1, ix, False)
            probs = probs.masked_fill(mask, 0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

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
        if eos_id >= 0 and token.item() == eos_id: break
    return x  # [1, T0+gen]

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--spm", type=str, default="tokenizer.model")
    ap.add_argument("--prompts", type=str, nargs="*", default=[
        "In a forgotten valley,",
        "Once upon a midnight,",
        "The old man whispered:",
        "She opened the letter and discovered",
        "In the beginning of winter,"
    ])
    ap.add_argument("--samples_per_prompt", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--eos_id", type=int, default=-1)
    ap.add_argument("--out_csv", type=str, default="gen_quality.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = find_ckpt(args.ckpt)
    print(f"[LOAD] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    margs = ModelArgs(**ckpt["args"])
    model = TransformerLM(margs).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    encode, decode, _ = load_tokenizer(args.spm)

    rows = []
    all_texts = []
    for p in args.prompts:
        pid = encode(p)
        start = torch.tensor([pid], dtype=torch.long, device=device)
        for s in range(args.samples_per_prompt):
            out = generate_ids(model, start,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature,
                               top_k=args.top_k, top_p=args.top_p,
                               eos_id=args.eos_id)
            gen_len = out.shape[1] - len(pid)
            gen_ids = out[0].tolist()
            text = decode(gen_ids)
            all_texts.append(text)

            # metrics on generated segment
            ppl, avg_entropy = conditional_ppl_and_entropy(model, out, gen_len)
            d1 = distinct_n(gen_ids[-gen_len:], 1)
            d2 = distinct_n(gen_ids[-gen_len:], 2)
            d3 = distinct_n(gen_ids[-gen_len:], 3)
            max_tri, rep_ratio = repetition_stats(gen_ids[-gen_len:])

            rows.append({
                "prompt": p.replace("\n", " "),
                "tokens_total": len(gen_ids),
                "tokens_generated": gen_len,
                "ppl_generated": round(ppl, 3),
                "avg_entropy_bits": round(avg_entropy, 3),
                "distinct1": round(d1, 3),
                "distinct2": round(d2, 3),
                "distinct3": round(d3, 3),
                "max_trigram_freq": max_tri,
                "repeat_ratio": round(rep_ratio, 3),
                "ends_sentence": text.strip().endswith(('.', '!', '?', '”', '"', '’'))
            })

    # write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary “quality gate”
    ppl_vals = [r["ppl_generated"] for r in rows]
    ent_vals = [r["avg_entropy_bits"] for r in rows]
    d2_vals = [r["distinct2"] for r in rows]
    rep_vals = [r["repeat_ratio"] for r in rows]
    eos_ok = eos_rate(all_texts)

    print("\n=== SUMMARY ===")
    print(f"samples: {len(rows)}  | prompts: {len(args.prompts)}  | max_new_tokens: {args.max_new_tokens}")
    print(f"ppl_generated:  median={stats.median(ppl_vals):.2f}  mean={stats.mean(ppl_vals):.2f}")
    print(f"avg_entropy_bits: median={stats.median(ent_vals):.2f} (2–6 bits is typical for non-greedy decoding at this scale)")
    print(f"distinct-2: median={stats.median(d2_vals):.3f}  (higher is more diverse; <0.3 usually means loops)")
    print(f"repeat_ratio: median={stats.median(rep_vals):.3f}  ( >0.4 often means bad repetition)")
    print(f"ends-with-sentence: {eos_ok*100:.1f}%")

    # crude gate (tune for your data scale)
    bad = 0
    if stats.median(ppl_vals) > 40: bad += 1
    if stats.median(d2_vals) < 0.30: bad += 1
    if stats.median(rep_vals) > 0.45: bad += 1
    verdict = "PASS" if bad == 0 else ("WARN" if bad == 1 else "FAIL")
    print(f"\nQUALITY GATE: {verdict}")
    print(f"[CSV] -> {args.out_csv}")

if __name__ == "__main__":
    main()
