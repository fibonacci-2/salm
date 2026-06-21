"""
train_bpe.py
============
Train a BPE tokenizer on pre-cleaned dialectal Arabic CSVs.

Assumes input CSVs are already clean — no filtering or text normalisation applied.
Reads text columns, writes a plain-text corpus, trains BPE, saves tokenizer.

Usage:
    python train_bpe.py \
        --files data/reddit.csv data/youtube.csv \
        --text_cols body comment_text \
        --output tokenizer_da/ \
        --vocab_size 32000

Dependencies:
    pip install tokenizers pandas tqdm
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece as WordPieceDecoder


# ─────────────────────────────────────────────────────────────────
# Special tokens
# ─────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<mask>",
]


# ─────────────────────────────────────────────────────────────────
# 1.  Build corpus file from CSVs
# ─────────────────────────────────────────────────────────────────

def build_corpus(
    csv_paths: list[str],
    text_cols: list[str],
    output_dir: Path,
    chunksize: int = 100_000,
) -> Path:
    """
    Read text columns from each CSV and write one sentence per line
    to a plain-text corpus file for the tokenizer trainer.

    Args:
        csv_paths  : list of CSV file paths
        text_cols  : text column names to extract, tried in order per file
        output_dir : where to write corpus.txt
        chunksize  : pandas read chunk size (rows)

    Returns:
        Path to the written corpus file
    """
    corpus_path = output_dir / "corpus.txt"
    total_written = 0

    with open(corpus_path, "w", encoding="utf-8") as out:
        for csv_path in csv_paths:
            print(f"Reading: {csv_path}")
            file_count = 0

            for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):

                # pick the first matching text column in this file
                col = next((c for c in text_cols if c in chunk.columns), None)
                if col is None:
                    raise ValueError(
                        f"None of {text_cols} found in {csv_path}. "
                        f"Columns present: {list(chunk.columns)}"
                    )

                for text in chunk[col].dropna():
                    line = str(text).strip()
                    if line:
                        out.write(line + "\n")
                        file_count += 1

            print(f"  → {file_count:,} lines written from {Path(csv_path).name}")
            total_written += file_count

    print(f"\nCorpus total : {total_written:,} lines")
    print(f"Corpus path  : {corpus_path}\n")
    return corpus_path


# ─────────────────────────────────────────────────────────────────
# 2.  Train BPE tokenizer
# ─────────────────────────────────────────────────────────────────

def train_tokenizer(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    continuing_subword_prefix: str = "##",
) -> Tokenizer:
    """
    Train BPE from a plain-text corpus and save to output_dir.

    Design choices:
      - Whitespace pre-tokenizer: correct for Arabic (space-delimited, no ByteLevel distortion)
      - No normalizer: data is already clean; normalising here risks stripping dialectal chars
      - ## subword prefix: marks non-initial pieces, useful for token-classification fine-tuning
      - WordPiece decoder: re-inserts spaces before non-## tokens; correct pair for Whitespace
        pre-tokenizer. BPEDecoder is wrong here — it expects ByteLevel Ġ-encoded spaces.

    Args:
        corpus_path               : plain-text file, one document/sentence per line
        output_dir                : directory to save tokenizer.json
        vocab_size                : target vocabulary size (32k default, 64k for larger corpora)
        min_frequency             : minimum pair frequency to merge
        continuing_subword_prefix : prefix for non-initial subword pieces

    Returns:
        Trained Tokenizer object
    """
    tokenizer = Tokenizer(BPE(
        unk_token="<unk>",
        continuing_subword_prefix=continuing_subword_prefix,
    ))

    # Whitespace splitting only — no ByteLevel, which corrupts Arabic codepoints
    tokenizer.pre_tokenizer = Whitespace()

    # WordPiece decoder re-inserts spaces before every non-## token,
    # which is correct for a Whitespace pre-tokenizer.
    # BPEDecoder is wrong here — it expects ByteLevel-encoded spaces (Ġ).
    tokenizer.decoder = WordPieceDecoder(prefix="##")

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix=continuing_subword_prefix,
        show_progress=True,
    )

    print(f"Training BPE tokenizer …")
    print(f"  vocab_size   : {vocab_size:,}")
    print(f"  min_frequency: {min_frequency}")
    print(f"  corpus       : {corpus_path}\n")

    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    # ── save ──
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\nTokenizer saved → {tokenizer_path}")

    return tokenizer


# ─────────────────────────────────────────────────────────────────
# 3.  Evaluation
# ─────────────────────────────────────────────────────────────────

ARABIC_RE   = re.compile(r"[\u0600-\u06FF]")
DIACRITIC_RE = re.compile(r"[\u064B-\u065F\u0670]")

EVAL_SAMPLES = [
    # Egyptian — no punctuation, clean round-trip baseline
    ("EGY", "مش عارف ليه بس احساسي كده وخلاص"),
    # Levantine — no punctuation
    ("LEV", "شو بتحكي هاد مو منطق والله"),
    # Gulf — contains ة (taa marbuta), tests rare char handling
    ("GLF", "والله ما ادري وش السالفة بس ابد"),
    # Maghrebi
    ("MGH", "واش راك لباس إن شاء الله كيفاش حالك"),
    # Iraqi
    ("IRQ", "شكو ماكو كلشي تمام ابو علي"),
    # Arabizi (Lebanese) — expected all-<unk>, corpus is Arabic-script only
    ("ARA", "haha 3njad ma 3raft chou sar shou hal mawdo3"),
    # Code-switch Arabic-English
    ("MIX", "الموضوع literally مش منطقي خالص whatever"),
]


def evaluate_tokenizer(tokenizer: Tokenizer) -> None:
    """
    Print per-sample tokenisation results and aggregate vocabulary stats.
    """
    print("\n── Sample tokenisations ────────────────────────────────────────")
    fertilities = []   # tokens per character (lower = better compression)

    for label, text in EVAL_SAMPLES:
        enc = tokenizer.encode(text)
        unk_id   = tokenizer.token_to_id("<unk>")
        unk_count = enc.ids.count(unk_id)
        fertility = len(enc.tokens) / max(len(text), 1)
        fertilities.append(fertility)
        print(f"  [{label}] {text}")
        print(f"         tokens ({len(enc.tokens)}, unk={unk_count}): {enc.tokens}")
        print(f"         fertility: {fertility:.3f} tok/char")
        print()

    avg_fertility = sum(fertilities) / len(fertilities)

    # ── vocabulary stats ──
    vocab = tokenizer.get_vocab()
    arabic_tokens    = [t for t in vocab if ARABIC_RE.search(t)]
    diacritic_tokens = [t for t in vocab if DIACRITIC_RE.search(t)]
    subword_tokens   = [t for t in vocab if t.startswith("##")]
    latin_tokens     = [t for t in vocab if re.search(r"[a-zA-Z]", t)]

    print("── Vocabulary statistics ───────────────────────────────────────")
    print(f"  Total vocab size      : {len(vocab):>8,}")
    print(f"  Arabic tokens         : {len(arabic_tokens):>8,}  ({100*len(arabic_tokens)/len(vocab):.1f}%)")
    print(f"  Subword (##) tokens   : {len(subword_tokens):>8,}  ({100*len(subword_tokens)/len(vocab):.1f}%)")
    print(f"  Latin / Arabizi tokens: {len(latin_tokens):>8,}  ({100*len(latin_tokens)/len(vocab):.1f}%)")
    print(f"  Diacritic tokens      : {len(diacritic_tokens):>8,}  ← should be ~0")
    print(f"  Avg fertility         : {avg_fertility:.3f} tok/char  ← lower is better compression")

    # ── special token IDs ──
    print("\n── Special token IDs ───────────────────────────────────────────")
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        print(f"  {tok:<20} → {tid}")

    # ── round-trip check ──
    # Mismatches caused purely by <unk> are expected — the char was never in vocab.
    # True decode bugs produce space-collapse or ## leakage with no <unk> involved.
    print("\n── Round-trip decode check ─────────────────────────────────────")
    unk_id = tokenizer.token_to_id("<unk>")
    for label, text in EVAL_SAMPLES[:5]:   # skip ARA/MIX — known all-unk
        enc     = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids)
        has_unk = unk_id in enc.ids
        match   = decoded.strip() == text.strip()

        if match:
            print(f"  ✓ [{label}]")
        elif has_unk:
            n_unk = enc.ids.count(unk_id)
            print(f"  ~ [{label}]  ({n_unk} <unk> — char not in vocab, expected loss)")
            print(f"    original : {repr(text.strip())}")
            print(f"    decoded  : {repr(decoded.strip())}")
        else:
            print(f"  ✗ [{label}]  DECODE ERROR — no <unk> involved, investigate")
            print(f"    original : {repr(text.strip())}")
            print(f"    decoded  : {repr(decoded.strip())}")


# ─────────────────────────────────────────────────────────────────
# 4.  CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BPE tokenizer on pre-cleaned dialectal Arabic CSVs"
    )
    p.add_argument(
        "--files", nargs="+", required=True,
        help="One or more CSV file paths (e.g. --files reddit.csv youtube.csv)"
    )
    p.add_argument(
        "--text_cols", nargs="+", default=["body", "comment_text", "text", "selftext"],
        help="Text column names to try (in order) per file"
    )
    p.add_argument(
        "--output", type=str, default="tokenizer_da/",
        help="Output directory"
    )
    p.add_argument("--vocab_size",  type=int, default=32_000)
    p.add_argument("--min_freq",    type=int, default=2,
                   help="Minimum BPE merge frequency")
    p.add_argument("--chunksize",   type=int, default=100_000,
                   help="Pandas CSV read chunk size")
    p.add_argument("--skip_corpus", action="store_true",
                   help="Reuse existing corpus.txt (skips CSV reading)")
    p.add_argument("--eval_only",   action="store_true",
                   help="Load existing tokenizer.json and run evaluation only")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    corpus_path    = outdir / "corpus.txt"
    tokenizer_path = outdir / "tokenizer.json"

    # ── eval-only mode ──
    if args.eval_only:
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"No tokenizer found at {tokenizer_path}")
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        evaluate_tokenizer(tokenizer)
        return

    # ── build corpus ──
    if args.skip_corpus and corpus_path.exists():
        print(f"Reusing existing corpus: {corpus_path}")
    else:
        corpus_path = build_corpus(
            csv_paths=args.files,
            text_cols=args.text_cols,
            output_dir=outdir,
            chunksize=args.chunksize,
        )

    # ── train ──
    tokenizer = train_tokenizer(
        corpus_path=corpus_path,
        output_dir=outdir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
    )

    # ── evaluate ──
    evaluate_tokenizer(tokenizer)

    print(f"\n✓ Done. Output in: {outdir}")
    print("""
Load later:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("tokenizer_da/tokenizer.json")

    # or as a HuggingFace fast tokenizer:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file="tokenizer_da/tokenizer.json")
""")


if __name__ == "__main__":
    main()