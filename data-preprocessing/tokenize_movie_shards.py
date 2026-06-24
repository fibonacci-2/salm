"""
tokenize_shards_movies.py
=========================
Tokenize a pre-cleaned movie transcript corpus (.txt) using the locally
trained BPE tokenizer and write fixed-size binary shards (.npy) for pretraining.

Input format (produced by the VTT cleaning script):
    One movie per paragraph, paragraphs separated by \\n\\n.

Shard layout:
    <output_dir>/
        movies_val_000000.npy       ← first shard = validation
        movies_train_000001.npy
        movies_train_000002.npy
        ...

Each .npy file is a flat uint32 array of token IDs.
BOS is prepended to every document (movie); EOS is appended.

Usage:
    python tokenize_shards_movies.py \\
        --corpus    data/salm/corpus.txt \\
        --tokenizer tokenizer/tokenizer.json \\
        --output    data/shards/ \\
        --shard_size 100_000_000 \\
        --nprocs 24

Dependencies:
    pip install tokenizers numpy tqdm
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
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
# TXT → token shards
# ─────────────────────────────────────────────────────────────────

def process_source(
    txt_path: str,
    source_tag: str,
    output_dir: Path,
    tokenizer_path: str,
    shard_size: int,
    nprocs: int,
    mp_chunksize: int = 64,
) -> int:
    """
    Read a plain-text corpus file (\\n\\n-separated movie documents),
    tokenize with a multiprocessing pool, and write fixed-size shards.

    Args:
        txt_path       : path to the .txt corpus file
        source_tag     : prefix for shard filenames, e.g. 'movies'
        output_dir     : directory to write .npy shards
        tokenizer_path : path to tokenizer.json
        shard_size     : tokens per shard (last shard holds remainder)
        nprocs         : worker processes
        mp_chunksize   : documents per imap task
    """
    print(f"\n{'─'*60}")
    print(f"Source  : {txt_path}")
    print(f"Tag     : {source_tag}")
    print(f"Shards  : {output_dir}")
    print(f"{'─'*60}")

    # ── read and split on double newline ──
    raw   = Path(txt_path).expanduser().read_text(encoding="utf-8", errors="ignore")
    texts = [" ".join(doc.splitlines()).strip()
             for doc in raw.split("\n\n")
             if doc.strip()]

    total_docs = len(texts)
    print(f"Documents: {total_docs:,}")

    # ── multiprocessing pool ──
    shard_index  = 0
    buf          = np.empty((shard_size,), dtype=np.uint32)
    token_count  = 0
    total_tokens = 0
    pbar         = None

    with mp.Pool(
        processes   = nprocs,
        initializer = worker_init,
        initargs    = (tokenizer_path,),
    ) as pool:
        for tokens in pool.imap(tokenize_doc, texts, chunksize=mp_chunksize):
            if pbar is None:
                split = "val" if shard_index == 0 else "train"
                pbar  = tqdm(
                    total = shard_size,
                    unit  = "tok",
                    desc  = f"  {source_tag} shard {shard_index:06d} [{split}]",
                )

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
                    pbar = tqdm(
                        total = shard_size,
                        unit  = "tok",
                        desc  = f"  {source_tag} shard {shard_index:06d} [train]",
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
        description="Tokenize movie transcript corpus (.txt) into pretraining shards"
    )
    p.add_argument("--corpus",      type=str, required=True,
                   help="Path to plain-text corpus file (\\n\\n-separated movies)")
    p.add_argument("--tokenizer",   type=str, required=True,
                   help="Path to tokenizer.json")
    p.add_argument("--output",      type=str, default="../data/pretraining/youtube-movies",
                   help="Output directory for .npy shard files")
    p.add_argument("--tag",         type=str, default="movies",
                   help="Prefix for shard filenames (default: movies)")
    p.add_argument("--shard_size",  type=int, default=int(1e7),
                   help="Tokens per shard (default: 100M)")
    p.add_argument("--nprocs",      type=int, default=mp.cpu_count(),
                   help="Number of worker processes")
    p.add_argument("--mp_chunk",    type=int, default=64,
                   help="Documents per imap task")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer_path = str(Path(args.tokenizer).resolve())
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    corpus_path = Path(args.corpus).expanduser()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer loaded  : {tokenizer_path}")
    print(f"Vocab size        : {tok.get_vocab_size():,}")
    print(f"Shard size        : {args.shard_size:,} tokens")
    print(f"Worker processes  : {args.nprocs}")
    print(f"Output directory  : {output_dir}")

    total = process_source(
        txt_path       = str(corpus_path),
        source_tag     = args.tag,
        output_dir     = output_dir,
        tokenizer_path = tokenizer_path,
        shard_size     = args.shard_size,
        nprocs         = args.nprocs,
        mp_chunksize   = args.mp_chunk,
    )

    print(f"\n{'═'*60}")
    print(f"Done.")
    print(f"Total tokens : {total:,}")
    print(f"Shards in    : {output_dir}")
    print(f"{'═'*60}")

    shards = sorted(output_dir.glob(f"{args.tag}_*.npy"))
    print(f"\nShard manifest ({len(shards)} files):")
    for s in shards:
        size_mb = s.stat().st_size / 1024 / 1024
        print(f"  {s.name:<45}  {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()