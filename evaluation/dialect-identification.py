"""
eval_dialect_id.py
==================
Evaluate GPT pretraining checkpoints on MADAR-26 dialect identification
using a linear probe (frozen backbone + trained classification head).

MADAR TSV format expected:
    sentID.BTEC  ALE  ALX  ALG  AMM  ASW  BAG  BAS  BEI  BEN  CAI  DAM
    DOH  FES  JED  JER  KHA  MSA  MOS  MUS  RAB  RIY  SAL  SAN  SFX  TRI  TUN

Label = city with highest score per row.

Usage:
    python eval_dialect_id.py \\
        --madar       data/madar.tsv \\
        --checkpoints outputs/log/ \\
        --tokenizer   tokenizer/tokenizer.json \\
        --output      results/dialect_id.csv \\
        --epochs      10 \\
        --batch_size  64

Dependencies:
    pip install torch tokenizers pandas numpy scikit-learn tqdm
"""

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tokenizers import Tokenizer
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# MADAR-26 city labels (column order matches the TSV header)
# ─────────────────────────────────────────────────────────────────

MADAR_CITIES = [
    "ALE", "ALX", "ALG", "AMM", "ASW", "BAG", "BAS", "BEI", "BEN",
    "CAI", "DAM", "DOH", "FES", "JED", "JER", "KHA", "MSA", "MOS",
    "MUS", "RAB", "RIY", "SAL", "SAN", "SFX", "TRI", "TUN",
]

# Coarse region groupings for region-level accuracy
CITY_TO_REGION = {
    "ALE": "LEV", "DAM": "LEV", "BEI": "LEV", "JER": "LEV", "AMM": "LEV",
    "CAI": "EGY", "ALX": "EGY", "ASW": "EGY",
    "BAG": "IRQ", "BAS": "IRQ", "MOS": "IRQ",
    "RIY": "GLF", "JED": "GLF", "DOH": "GLF", "KHA": "GLF",
    "MUS": "GLF", "SAL": "GLF",
    "SAN": "YEM",
    "TRI": "MGH", "TUN": "MGH", "SFX": "MGH",
    "ALG": "MGH", "FES": "MGH", "RAB": "MGH",
    "BEN": "MGH",
    "MSA": "MSA",
}


# ─────────────────────────────────────────────────────────────────
# GPT model (must match architecture used during pretraining)
# ─────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos     = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x  # (B, T, n_embd) — no lm_head; we use this as encoder

    def get_sentence_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool the last hidden states over *non-padding* token positions only.
        Padding token id is 0 (<pad>); those positions are excluded from the mean.
        Returns shape (B, n_embd).
        """
        hidden = self.forward(idx)                        # (B, T, n_embd)
        mask   = (idx != 0).unsqueeze(-1).float()         # (B, T, 1)  1=real, 0=pad
        summed = (hidden * mask).sum(dim=1)               # (B, n_embd)
        counts = mask.sum(dim=1).clamp(min=1)             # (B, 1)
        return summed / counts                            # (B, n_embd)


# ─────────────────────────────────────────────────────────────────
# Linear probe classifier
# ─────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, n_embd: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_embd, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class MADARDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int],
                 tokenizer: Tokenizer, max_len: int = 128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer.encode(self.texts[idx])
        ids = enc.ids[:self.max_len]

        # pad to max_len
        pad_len = self.max_len - len(ids)
        ids     = ids + [0] * pad_len   # 0 = <pad>

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_madar(tsv_path: str) -> tuple[list[str], list[str]]:
    """
    Read MADAR TSV and return (texts, city_labels).
    Label per row = city column with the highest score.
    """
    df = pd.read_csv(tsv_path, sep='\t')

    # The first column is sentID.BTEC (sentence text), rest are city scores
    text_col   = df.columns[0]
    city_cols  = [c for c in df.columns[1:] if c in MADAR_CITIES]

    missing = [c for c in MADAR_CITIES if c not in df.columns]
    if missing:
        print(f"  Warning: city columns not found in TSV: {missing}")

    texts  = df[text_col].astype(str).tolist()
    labels = df[city_cols].idxmax(axis=1).tolist()   # city with highest score

    label_counts = pd.Series(labels).value_counts()
    print(f"  Loaded {len(texts):,} sentences across {len(label_counts)} cities")
    print(f"  Label distribution:\n{label_counts.to_string()}\n")

    return texts, labels


# ─────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: GPT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract masked mean-pooled embeddings from frozen backbone, L2-normalized."""
    model.eval()
    all_embs, all_labels = [], []

    for ids, labels in tqdm(loader, desc="  Extracting embeddings", leave=False):
        ids = ids.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            emb = model.get_sentence_embedding(ids)          # (B, n_embd)
        emb = emb.float()
        # L2 normalize: puts all embeddings on unit sphere so linear probe
        # is scale-invariant and converges faster
        emb = F.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────
# Linear probe training
# ─────────────────────────────────────────────────────────────────

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    n_embd: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    lr: float = 1e-2,        # higher LR: embeddings are L2-normed so scale is known
) -> dict:
    """
    Train a linear probe on pre-extracted L2-normalized embeddings.

    Key fixes vs original:
      - class-weighted CrossEntropyLoss to counter majority-class collapse
      - higher default LR (1e-2 vs 1e-3) appropriate for normalized features
      - majority-class baseline reported for comparison
    """
    # ── majority class baseline ──
    from collections import Counter
    majority_class = Counter(y_train).most_common(1)[0][0]
    majority_acc   = (y_test == majority_class).mean()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    tr_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    # class weights: inverse frequency, so minority cities aren't ignored
    counts      = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts      = np.where(counts == 0, 1, counts)     # avoid div-by-zero
    weights     = torch.tensor(1.0 / counts).to(device)
    weights     = weights / weights.sum() * n_classes   # normalize

    probe     = LinearProbe(n_embd, n_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_acc   = 0.0
    best_state = None

    for epoch in range(epochs):
        probe.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            logits = probe(X_te.to(device))
            preds  = logits.argmax(dim=-1).cpu()
            acc    = (preds == y_te).float().mean().item()

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        logits = probe(X_te.to(device))
        preds  = logits.argmax(dim=-1).cpu().numpy()

    city_acc = (preds == y_test).mean()
    print(f"    Majority baseline : {majority_acc:.4f} ({majority_acc*100:.1f}%)")
    print(f"    Probe accuracy    : {city_acc:.4f} ({city_acc*100:.1f}%)  "
          f"[+{(city_acc - majority_acc)*100:.1f}pp vs baseline]")

    return {
        "city_acc"       : city_acc,
        "majority_acc"   : majority_acc,
        "preds"          : preds,
        "targets"        : y_test,
    }


# ─────────────────────────────────────────────────────────────────
# Per-checkpoint evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    y_train: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    args,
    device: torch.device,
) -> dict:
    """Load checkpoint, extract embeddings, train probe, return metrics."""

    # ── load model ──
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # use config from checkpoint if available, else default
    config = ckpt.get('config', GPTConfig())
    if not isinstance(config, GPTConfig):
        # config saved as dict in some versions
        config = GPTConfig(**{k: v for k, v in vars(config).items()
                              if k in GPTConfig.__dataclass_fields__})

    model = GPT(config)

    # strip DDP wrapper prefix if present
    state = ckpt['model']
    state = {k.replace('_orig_mod.', '').replace('module.', ''): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)

    step = ckpt.get('step', Path(ckpt_path).stem)

    print(f"\n{'─'*60}")
    print(f"Checkpoint : {Path(ckpt_path).name}  (step {step})")

    # ── extract embeddings ──
    X_train, _ = extract_embeddings(model, train_loader, device)
    X_test,  _ = extract_embeddings(model, test_loader,  device)

    n_embd    = config.n_embd
    n_classes = len(le.classes_)

    # ── train probe ──
    print(f"  Training linear probe ({epochs} epochs, {n_classes} classes) …")
    metrics = train_linear_probe(
        X_train, y_train,
        X_test,  y_test,
        n_classes = n_classes,
        n_embd    = n_embd,
        epochs    = args.epochs,
        batch_size= args.batch_size,
        device    = device,
    )

    # ── region-level accuracy ──
    pred_cities   = le.inverse_transform(metrics['preds'])
    target_cities = le.inverse_transform(metrics['targets'])
    pred_regions  = [CITY_TO_REGION.get(c, "UNK") for c in pred_cities]
    tgt_regions   = [CITY_TO_REGION.get(c, "UNK") for c in target_cities]
    region_acc    = np.mean([p == t for p, t in zip(pred_regions, tgt_regions)])

    print(f"  City-level  accuracy : {metrics['city_acc']:.4f}  ({metrics['city_acc']*100:.1f}%)")
    print(f"  Region-level accuracy: {region_acc:.4f}  ({region_acc*100:.1f}%)")

    # ── per-city breakdown ──
    report = classification_report(
        target_cities, pred_cities,
        labels=sorted(set(target_cities)),
        zero_division=0,
        output_dict=True,
    )
    per_city_f1 = {city: report[city]['f1-score']
                   for city in sorted(set(target_cities))
                   if city in report}

    return {
        "checkpoint"   : Path(ckpt_path).name,
        "step"         : step,
        "city_acc"     : metrics['city_acc'],
        "majority_acc" : metrics['majority_acc'],
        "region_acc"   : region_acc,
        **{f"f1_{city}": f1 for city, f1 in per_city_f1.items()},
    }


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate GPT checkpoints on MADAR-26 dialect ID via linear probe"
    )
    p.add_argument("--madar",       default="../data/eval/madar_combined.tsv",
                   help="Path to MADAR TSV file")
    p.add_argument("--checkpoints", default="../outputs/salm-large/log",
                   help="Directory containing model_*.pt checkpoint files")
    p.add_argument("--tokenizer",   default="../tokenizer/tokenizer/tokenizer.json",
                   help="Path to tokenizer.json")
    p.add_argument("--output",      default="da-results-salm-large.csv",
                   help="Path to write results CSV")
    p.add_argument("--epochs",      type=int, default=10,
                   help="Linear probe training epochs")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--max_len",     type=int, default=128,
                   help="Max token length per sentence")
    p.add_argument("--test_size",   type=float, default=0.2,
                   help="Fraction of data held out for testing")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      type=str, default=None,
                   help="cuda / cpu (auto-detected if not set)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    global epochs   # used inside evaluate_checkpoint
    args   = parse_args()
    epochs = args.epochs

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device     : {device}")

    # ── load and encode MADAR ──
    print(f"\nLoading MADAR: {args.madar}")
    texts, city_labels = load_madar(args.madar)

    le      = LabelEncoder()
    y       = le.fit_transform(city_labels)
    n_classes = len(le.classes_)
    print(f"Classes ({n_classes}): {list(le.classes_)}")

    # ── train/test split (stratified) ──
    X_tr_txt, X_te_txt, y_train, y_test = train_test_split(
        texts, y,
        test_size    = args.test_size,
        stratify     = y,
        random_state = args.seed,
    )
    print(f"Train: {len(X_tr_txt):,}  |  Test: {len(X_te_txt):,}")

    # ── tokenizer ──
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # ── datasets & loaders ──
    tr_ds = MADARDataset(X_tr_txt, y_train, tokenizer, max_len=args.max_len)
    te_ds = MADARDataset(X_te_txt, y_test,  tokenizer, max_len=args.max_len)

    train_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(te_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── discover checkpoints ──
    ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoints, "model_*.pt")))
    assert len(ckpt_paths) > 0, \
        f"No model_*.pt files found in {args.checkpoints}"
    print(f"\nCheckpoints found: {len(ckpt_paths)}")
    for p in ckpt_paths:
        print(f"  {Path(p).name}")

    # ── evaluate each checkpoint ──
    all_results = []
    for ckpt_path in ckpt_paths:
        result = evaluate_checkpoint(
            ckpt_path   = ckpt_path,
            train_loader= train_loader,
            test_loader = test_loader,
            y_train     = y_train,
            y_test      = y_test,
            le          = le,
            args        = args,
            device      = device,
        )
        all_results.append(result)

    # ── results table ──
    df = pd.DataFrame(all_results).sort_values("step")

    print(f"\n{'═'*60}")
    print("Results summary:")
    print(df[["checkpoint", "step", "majority_acc", "city_acc", "region_acc"]].to_string(index=False))

    best = df.loc[df["city_acc"].idxmax()]
    print(f"\nBest checkpoint : {best['checkpoint']}")
    print(f"  Majority baseline : {best['majority_acc']*100:.1f}%")
    print(f"  City accuracy     : {best['city_acc']*100:.1f}%")
    print(f"  Region accuracy   : {best['region_acc']*100:.1f}%")

    # ── save CSV ──
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nFull results saved to: {args.output}")

    # ── per-city F1 for best checkpoint ──
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    if f1_cols:
        print(f"\nPer-city F1 for best checkpoint ({best['checkpoint']}):")
        f1_vals = best[f1_cols].sort_values(ascending=False)
        for col, val in f1_vals.items():
            city = col.replace("f1_", "")
            region = CITY_TO_REGION.get(city, "?")
            print(f"  {city} ({region}):  {val:.3f}")


if __name__ == "__main__":
    main()