"""
tokenize_shards_hf.py
=====================
Load a HuggingFace dataset, apply basic Arabic filtering, and write
fixed-size pretraining shards (.npy) using the local BPE tokenizer.

Filtering applied before tokenization (consistent with Reddit/YouTube pipeline):
  - Minimum character length
  - Arabic content gate (≥ threshold fraction of chars must be Arabic script)
  - Punctuation removal (Latin chars, non-Arabic punctuation, diacritics)

Shard layout:
    <output_dir>/
        <tag>_val_000000.npy
        <tag>_train_000001.npy
        ...

Usage:
    python tokenize_shards_hf.py \\
        --dataset    ClusterlabAi/101_billion_arabic_words_dataset \\
        --split      train \\
        --text_col   text \\
        --tokenizer  tokenizer/tokenizer.json \\
        --output     data/shards/ \\
        --tag        arabic101b \\
        --shard_size 100_000_000 \\
        --nprocs     24

    # stream a large dataset without downloading it fully:
    python tokenize_shards_hf.py \\
        --dataset    Omartificial-Intelligence-Space/FineWeb2-MSA \\
        --split      train \\
        --tokenizer  tokenizer/tokenizer.json \\
        --output     data/shards/ \\
        --tag        fineweb_msa \\
        --streaming

Dependencies:
    pip install tokenizers datasets numpy tqdm
"""

import argparse
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer


# ─────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────

_ARABIC_RE = re.compile(r'[\u0620-\u06B0\u06BA-\u06BF\u06CC-\u06D3\u0750-\u077F\u08A0-\u08FF]')


def filter_and_clean(text: str, min_chars: int, arabic_threshold: float) -> str | None:
    """
    Apply pre-tokenization filtering and cleaning.

    Steps (consistent with Reddit/YouTube and movie pipelines):
      1. Minimum length gate
      2. Remove Latin characters
      3. Remove punctuation — keep only Arabic letters, Arabic digits, spaces
      4. Remove diacritics (tashkeel)
      5. Normalize whitespace
      6. Arabic content gate
      7. Post-cleaning minimum length gate

    Returns cleaned text, or None if the document should be discarded.
    """
    if not isinstance(text, str) or len(text) < min_chars:
        return None

    # 1. Remove Latin
    text = re.sub(r'[a-zA-Z]+', ' ', text)
    # 2. Keep only Arabic letters, Arabic-Indic digits (٠-٩), spaces
    text = re.sub(
        r'[^\u0620-\u06B0\u06BA-\u06BF\u06CC-\u06D3'
        r'\u0660-\u0669\u0750-\u077F\u08A0-\u08FF\s]',
        ' ', text
    )
    # 3. Remove tashkeel / diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < min_chars:
        return None

    # 5. Arabic content gate
    arabic_chars = len(_ARABIC_RE.findall(text))
    if arabic_chars / max(len(text.replace(' ', '')), 1) < arabic_threshold:
        return None

    return text


# ─────────────────────────────────────────────────────────────────
# Worker (tokenizer loaded once per process)
# ─────────────────────────────────────────────────────────────────

_tokenizer = None
_bos_id    = None
_eos_id    = None


def worker_init(tokenizer_path: str) -> None:
    global _tokenizer, _bos_id, _eos_id
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    _bos_id    = _tokenizer.token_to_id("<bos>")
    _eos_id    = _tokenizer.token_to_id("<eos>")
    assert _bos_id is not None, "<bos> not found in vocab"
    assert _eos_id is not None, "<eos> not found in vocab"


def tokenize_doc(text: str) -> np.ndarray | None:
    """Tokenize a single pre-cleaned document. Returns None for empty results."""
    enc = _tokenizer.encode(text)
    if not enc.ids:
        return None
    return np.array([_bos_id] + enc.ids + [_eos_id], dtype=np.uint32)


def write_shard(path: str, buf: np.ndarray, count: int) -> None:
    np.save(path, buf[:count])


# ─────────────────────────────────────────────────────────────────
# Core sharding loop
# ─────────────────────────────────────────────────────────────────

def shard_documents(
    texts,                  # iterable of str (list or streaming generator)
    total: int | None,      # total doc count for tqdm (None if unknown/streaming)
    source_tag: str,
    output_dir: Path,
    tokenizer_path: str,
    shard_size: int,
    nprocs: int,
    min_chars: int,
    arabic_threshold: float,
    mp_chunksize: int = 64,
) -> int:
    """Tokenize an iterable of texts and write shards. Returns total token count."""

    print(f"\n{'─'*60}")
    print(f"Tag              : {source_tag}")
    print(f"Min chars        : {min_chars}")
    print(f"Arabic threshold : {arabic_threshold}")
    print(f"{'─'*60}")

    shard_index  = 0
    buf          = np.empty((shard_size,), dtype=np.uint32)
    token_count  = 0
    total_tokens = 0
    kept = 0
    dropped = 0
    pbar = None

    def _filtered(texts):
        """Pre-filter generator — runs in main process before pool."""
        for text in tqdm(texts, total=total, desc="  Filtering", unit="doc"):
            cleaned = filter_and_clean(text, min_chars, arabic_threshold)
            if cleaned:
                yield cleaned

    with mp.Pool(
        processes   = nprocs,
        initializer = worker_init,
        initargs    = (tokenizer_path,),
    ) as pool:
        for tokens in pool.imap(_tokenize_worker, _filtered(texts),
                                chunksize=mp_chunksize):
            if tokens is None:
                dropped += 1
                continue
            kept += 1

            if pbar is None:
                split = "val" if shard_index == 0 else "train"
                pbar  = tqdm(total=shard_size, unit="tok",
                             desc=f"  {source_tag} shard {shard_index:06d} [{split}]")

            while len(tokens) > 0:
                space = shard_size - token_count
                if len(tokens) <= space:
                    buf[token_count : token_count + len(tokens)] = tokens
                    token_count  += len(tokens)
                    total_tokens += len(tokens)
                    pbar.update(len(tokens))
                    tokens = tokens[0:0]
                else:
                    buf[token_count : token_count + space] = tokens[:space]
                    pbar.update(space)
                    pbar.close()

                    split    = "val" if shard_index == 0 else "train"
                    filename = output_dir / f"{source_tag}_{split}_{shard_index:06d}"
                    write_shard(str(filename), buf, shard_size)
                    print(f"  ✓ wrote {filename}.npy  ({shard_size:,} tokens)")

                    total_tokens += space
                    shard_index  += 1
                    token_count   = 0
                    tokens        = tokens[space:]
                    pbar = tqdm(total=shard_size, unit="tok",
                                desc=f"  {source_tag} shard {shard_index:06d} [train]")

        # flush final shard
        if token_count > 0:
            if pbar:
                pbar.close()
            split    = "val" if shard_index == 0 else "train"
            filename = output_dir / f"{source_tag}_{split}_{shard_index:06d}"
            write_shard(str(filename), buf, token_count)
            print(f"  ✓ wrote {filename}.npy  ({token_count:,} tokens)  [partial]")
            total_tokens += token_count

    print(f"\n  Kept    : {kept:,}  |  Dropped (filter/empty): {dropped:,}"
          f"  ({100*dropped/max(kept+dropped,1):.1f}% drop rate)")
    print(f"  Tokens  : {total_tokens:,}  across {shard_index + 1} shards")
    return total_tokens


def _tokenize_worker(text: str) -> np.ndarray | None:
    """Top-level function required for multiprocessing pickling."""
    return tokenize_doc(text)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize a HuggingFace dataset into pretraining shards"
    )
    p.add_argument("--dataset",    required=True,
                   help="HF dataset name, e.g. ClusterlabAi/101_billion_arabic_words_dataset")
    p.add_argument("--split",      default="train",
                   help="Dataset split (default: train)")
    p.add_argument("--text_col",   default="text",
                   help="Column containing document text (default: text)")
    p.add_argument("--subset",     default=None,
                   help="Dataset subset/config name if required")
    p.add_argument("--tokenizer",  required=True,
                   help="Path to tokenizer.json")
    p.add_argument("--output",     default="../data/pretraining/twitter",
                   help="Output directory for .npy shards")
    p.add_argument("--tag",        default=None,
                   help="Shard filename prefix (default: derived from dataset name)")
    p.add_argument("--shard_size", type=int, default=int(1e7),
                   help="Tokens per shard (default: 100M)")
    p.add_argument("--nprocs",     type=int, default=mp.cpu_count(),
                   help="Worker processes")
    p.add_argument("--mp_chunk",   type=int, default=64,
                   help="Documents per imap task")
    p.add_argument("--min_chars",  type=int, default=20,
                   help="Minimum character length after cleaning (default: 20)")
    p.add_argument("--arabic_threshold", type=float, default=0.4,
                   help="Min fraction of chars that must be Arabic (default: 0.4)")
    p.add_argument("--streaming",  action="store_true",
                   help="Use HF streaming mode (avoids full download)")
    p.add_argument("--limit",      type=int, default=None,
                   help="Process only the first N documents (for testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer_path = str(Path(args.tokenizer).resolve())
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # derive tag from dataset name if not given
    tag = args.tag or args.dataset.split("/")[-1].lower().replace("-", "_")[:20]

    tok = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer   : {tokenizer_path}  ({tok.get_vocab_size():,} vocab)")
    print(f"Dataset     : {args.dataset}  (split={args.split})")
    print(f"Tag         : {tag}")
    print(f"Shard size  : {args.shard_size:,} tokens")
    print(f"Workers     : {args.nprocs}")
    print(f"Streaming   : {args.streaming}")
    print(f"Min chars   : {args.min_chars}")
    print(f"Arabic thr  : {args.arabic_threshold}")

    # ── load dataset ──
    from datasets import load_dataset
    load_kwargs = dict(streaming=args.streaming)
    if args.subset:
        load_kwargs["name"] = args.subset

    ds = load_dataset(args.dataset, split=args.split, **load_kwargs)

    # extract text column as iterable
    if args.streaming:
        texts = (row[args.text_col] for row in ds)
        total = None   # unknown for streaming
    else:
        texts = ds[args.text_col]
        total = len(texts)
        print(f"Documents   : {total:,}")

    if args.limit:
        import itertools
        texts = itertools.islice(texts, args.limit)
        total = args.limit
        print(f"Limit       : {args.limit:,}")

    # ── shard ──
    grand_total = shard_documents(
        texts            = texts,
        total            = total,
        source_tag       = tag,
        output_dir       = output_dir,
        tokenizer_path   = tokenizer_path,
        shard_size       = args.shard_size,
        nprocs           = args.nprocs,
        min_chars        = args.min_chars,
        arabic_threshold = args.arabic_threshold,
        mp_chunksize     = args.mp_chunk,
    )

    print(f"\n{'═'*60}")
    print(f"Done. Grand total tokens : {grand_total:,}")
    print(f"Shards in               : {output_dir}")
    print(f"{'═'*60}")

    shards = sorted(output_dir.glob(f"{tag}_*.npy"))
    print(f"\nShard manifest ({len(shards)} files):")
    for s in shards:
        print(f"  {s.name:<45}  {s.stat().st_size/1024/1024:>8.1f} MB")


if __name__ == "__main__":
    main()