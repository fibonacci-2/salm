"""
tokenize_shards.py
==================
Tokenize pre-cleaned Reddit and YouTube CSV files using the locally trained
BPE tokenizer and write fixed-size binary shards (.npy) for pretraining.

Shard layout (mirrors nanoGPT / fineweb convention):
    <output_dir>/
        reddit_val_000000.npy       ← first shard = validation
        reddit_train_000001.npy
        reddit_train_000002.npy
        ...
        youtube_val_000000.npy
        youtube_train_000001.npy
        ...

Each .npy file is a flat uint32 array of token IDs.
BOS is prepended to every document; EOS is appended.

Usage:
    python tokenize_shards.py \
        --reddit  data/reddit.csv \
        --youtube data/youtube.csv \
        --text_cols text comment \
        --tokenizer tokenizer/tokenizer.json \
        --output  data/shards/ \
        --shard_size 100_000_000 \
        --nprocs 24

Dependencies:
    pip install tokenizers pandas numpy tqdm
"""

import argparse
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizers import Tokenizer


# ─────────────────────────────────────────────────────────────────
# Globals (populated in worker_init, safe for fork-based mp)
# ─────────────────────────────────────────────────────────────────
_tokenizer: Tokenizer | None = None
_bos_id: int | None = None
_eos_id: int | None = None


def worker_init(tokenizer_path: str) -> None:
    """Load tokenizer once per worker process."""
    global _tokenizer, _bos_id, _eos_id
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    _bos_id    = _tokenizer.token_to_id("<bos>")
    _eos_id    = _tokenizer.token_to_id("<eos>")
    assert _bos_id is not None, "<bos> not found in tokenizer vocab"
    assert _eos_id is not None, "<eos> not found in tokenizer vocab"


def tokenize_doc(text: str) -> np.ndarray:
    """
    Tokenize a single document string.
    Returns a uint32 numpy array: [<bos>, tok1, tok2, ..., tokN, <eos>]
    """
    enc = _tokenizer.encode(text)
    ids = [_bos_id] + enc.ids + [_eos_id]
    return np.array(ids, dtype=np.uint32)


# ─────────────────────────────────────────────────────────────────
# Shard writer
# ─────────────────────────────────────────────────────────────────

def write_shard(path: str, buf: np.ndarray, count: int) -> None:
    np.save(path, buf[:count])


# ─────────────────────────────────────────────────────────────────
# CSV → token shards
# ─────────────────────────────────────────────────────────────────

def process_source(
    csv_path: str,
    text_cols: list[str],
    source_tag: str,
    output_dir: Path,
    tokenizer_path: str,
    shard_size: int,
    nprocs: int,
    chunksize: int = 100_000,
    mp_chunksize: int = 64,
) -> None:
    """
    Read a single CSV, tokenize all rows with a multiprocessing pool,
    and write fixed-size shards to output_dir.

    Args:
        csv_path       : path to the CSV file
        text_cols      : column names to try in order (first match wins)
        source_tag     : prefix for shard filenames, e.g. 'reddit' or 'youtube'
        output_dir     : directory to write .npy shards
        tokenizer_path : path to tokenizer.json
        shard_size     : tokens per shard (last shard holds remainder)
        nprocs         : worker processes
        chunksize      : pandas CSV read chunk size (rows)
        mp_chunksize   : documents per imap task
    """
    print(f"\n{'─'*60}")
    print(f"Source  : {csv_path}")
    print(f"Tag     : {source_tag}")
    print(f"Shards  : {output_dir}")
    print(f"{'─'*60}")

    # ── stream all texts from CSV ──
    texts: list[str] = []
    col_used: str | None = None

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if col_used is None:
            col_used = next((c for c in text_cols if c in chunk.columns), None)
            if col_used is None:
                raise ValueError(
                    f"None of {text_cols} found in {csv_path}. "
                    f"Available: {list(chunk.columns)}"
                )
        texts.extend(chunk[col_used].dropna().astype(str).tolist())

    total_docs = len(texts)
    print(f"Documents: {total_docs:,}  |  column: '{col_used}'")

    # ── multiprocessing pool ──
    shard_index = 0
    buf         = np.empty((shard_size,), dtype=np.uint32)
    token_count = 0
    total_tokens = 0
    pbar        = None

    with mp.Pool(
        processes=nprocs,
        initializer=worker_init,
        initargs=(tokenizer_path,),
    ) as pool:
        for tokens in pool.imap(tokenize_doc, texts, chunksize=mp_chunksize):
            if pbar is None:
                split = "val" if shard_index == 0 else "train"
                pbar  = tqdm(
                    total=shard_size,
                    unit="tok",
                    desc=f"  {source_tag} shard {shard_index:06d} [{split}]",
                )

            while len(tokens) > 0:
                space = shard_size - token_count

                if len(tokens) <= space:
                    # fits entirely in current shard
                    buf[token_count : token_count + len(tokens)] = tokens
                    token_count  += len(tokens)
                    total_tokens += len(tokens)
                    pbar.update(len(tokens))
                    tokens = tokens[0:0]   # empty — exit while
                else:
                    # fill remainder of current shard, flush, start new one
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
                    pbar = tqdm(
                        total=shard_size,
                        unit="tok",
                        desc=f"  {source_tag} shard {shard_index:06d} [train]",
                    )

        # ── flush final partial shard ──
        if token_count > 0:
            if pbar:
                pbar.close()
            split    = "val" if shard_index == 0 else "train"
            filename = output_dir / f"{source_tag}_{split}_{shard_index:06d}"
            write_shard(str(filename), buf, token_count)
            print(f"  ✓ wrote {filename}.npy  ({token_count:,} tokens)  [partial — last shard]")
            total_tokens += token_count

    print(f"\n  {source_tag} done: {total_tokens:,} tokens across {shard_index + 1} shards")
    return total_tokens


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize dialectal Arabic CSVs into pretraining shards"
    )
    p.add_argument("--reddit",      type=str, default=None,
                   help="Path to Reddit CSV")
    p.add_argument("--youtube",     type=str, default=None,
                   help="Path to YouTube CSV")
    p.add_argument("--text_cols",   nargs="+",
                   default=["text", "body", "comment", "comment_text", "selftext"],
                   help="Text column names to try in order per file")
    p.add_argument("--tokenizer",   type=str, required=True,
                   help="Path to tokenizer.json")
    p.add_argument("--output",      type=str, default="data/shards/",
                   help="Output directory for .npy shard files")
    p.add_argument("--shard_size",  type=int, default=int(1e8),
                   help="Tokens per shard (default: 100M)")
    p.add_argument("--nprocs",      type=int, default=mp.cpu_count(),
                   help="Number of worker processes")
    p.add_argument("--csv_chunk",   type=int, default=100_000,
                   help="Pandas CSV read chunk size (rows)")
    p.add_argument("--mp_chunk",    type=int, default=64,
                   help="Documents per imap task")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.reddit and not args.youtube:
        raise ValueError("Provide at least one of --reddit or --youtube")

    tokenizer_path = str(Path(args.tokenizer).resolve())
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # quick vocab check
    tok = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer loaded  : {tokenizer_path}")
    print(f"Vocab size        : {tok.get_vocab_size():,}")
    print(f"Shard size        : {args.shard_size:,} tokens")
    print(f"Worker processes  : {args.nprocs}")
    print(f"Output directory  : {output_dir}")

    grand_total = 0

    sources = []
    if args.reddit:
        sources.append((args.reddit,  "reddit"))
    if args.youtube:
        sources.append((args.youtube, "youtube"))

    for csv_path, tag in sources:
        n = process_source(
            csv_path       = csv_path,
            text_cols      = args.text_cols,
            source_tag     = tag,
            output_dir     = output_dir,
            tokenizer_path = tokenizer_path,
            shard_size     = args.shard_size,
            nprocs         = args.nprocs,
            chunksize      = args.csv_chunk,
            mp_chunksize   = args.mp_chunk,
        )
        grand_total += n

    print(f"\n{'═'*60}")
    print(f"All sources done.")
    print(f"Grand total tokens : {grand_total:,}")
    print(f"Shards written to  : {output_dir}")
    print(f"{'═'*60}")

    # ── shard manifest ──
    shards = sorted(output_dir.glob("*.npy"))
    print(f"\nShard manifest ({len(shards)} files):")
    for s in shards:
        size_mb = s.stat().st_size / 1024 / 1024
        print(f"  {s.name:<45}  {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()