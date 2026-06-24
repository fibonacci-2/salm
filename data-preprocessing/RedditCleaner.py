"""
RedditCleaner — Arabic corpus cleaner for LLM pretraining.
Surgically removes English/non-Arabic content while preserving clean Arabic text.
"""

import re
import pandas as pd
from dataclasses import dataclass, field


# ── Unicode ranges ─────────────────────────────────────────────────────────────
ARABIC_BLOCK  = r'\u0600-\u06FF'
ARABIC_SUPP   = r'\u0750-\u077F'
ARABIC_EXT_A  = r'\u08A0-\u08FF'
ARABIC_PRES_A = r'\uFB50-\uFDFF'
ARABIC_PRES_B = r'\uFE70-\uFEFF'
ARABIC_MATH   = r'\U0001EE00-\U0001EEFF'

ARABIC_RE = re.compile(
    f'[{ARABIC_BLOCK}{ARABIC_SUPP}{ARABIC_EXT_A}{ARABIC_PRES_A}{ARABIC_PRES_B}{ARABIC_MATH}]'
)

# ── Quranic / Uthmanic special chars ──────────────────────────────────────────
QURAN_CHARS = re.compile(
    r'[\u0671'          # Alef Wasla ٱ
    r'\u0615'           # Sadaq sign
    r'\u0670'           # superscript alef
    r'\u06D6-\u06ED'    # Quranic annotation signs
    r'\u08D4-\u08FF'    # Arabic Extended-A marks
    r'\uFBB2-\uFBC1'    # Uthmanic stop marks
    r'\u0600-\u0605'    # Arabic number/anno signs
    r'\u0610-\u061A'    # Arabic extended signs
    r'\u06DD-\u06DF'    # End of Ayah, Rub signs
    r'\u06E0-\u06E4'    # Quranic marks
    r'\u06E7-\u06E8'
    r'\u06EA-\u06ED'
    r'ۭۚۖۗۘۙۛۜ'
    r']'
)

# ── Single Arabic letter (for spaced-text detection) ──────────────────────────
_SINGLE_AR = re.compile(rf'^[{ARABIC_BLOCK}{ARABIC_SUPP}{ARABIC_EXT_A}]$')

# ── All cleaning patterns ──────────────────────────────────────────────────────
PATTERNS = {
    'url':          re.compile(r'https?://\S+|www\.\S+', re.I),
    'mention':      re.compile(r'@\w+'),
    'subreddit':    re.compile(r'r/\w+|u/\w+'),
    'hashtag':      re.compile(r'#\w+'),
    # Handles normal &amp; &#200; AND spaced variants like & # 2 0 0 ;
    'html_entity':  re.compile(r'&\s*#?\s*[a-zA-Z0-9][\sa-zA-Z0-9]*\s*;', re.I),
    'html_tag':     re.compile(r'<[^>]+>'),
    'emoji':        re.compile(
                        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
                        r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
                        r'\U00002700-\U000027BF\U0001F900-\U0001F9FF'
                        r'\U00002600-\U000026FF]+', re.UNICODE),
    # Divider lines: _ _ _ _ _   or  -------  or ======  (3+ with optional spaces)
    'divider':      re.compile(r'(?:[-_=*~#][ \t]*){3,}'),
    # Runs of 3+ punctuation chars optionally spaced: . . .  or , , ,
    'spaced_punct': re.compile(r'(?:[^\w\s\u0600-\u06FF][ \t]*){3,}'),
    'repeated_char':  re.compile(r'(.)\1{3,}'),
    'repeated_punct': re.compile(r'([^\w\s\u0600-\u06FF])\1{2,}'),
    # Newline normalisation
    'extra_newline':  re.compile(r'[ \t]*\n[ \t]*\n+[ \t]*'),   # blank lines → space
    'newline':        re.compile(r'\n'),                         # lone \n → space
    'extra_space':    re.compile(r'[ \t]{2,}'),
    'english_word':   re.compile(r'\b[A-Za-z][A-Za-z0-9\'-]*\b'),
    'number_only':    re.compile(r'^\s*[\d\s\.,\-\+\(\)\u0660-\u0669]+\s*$'),
    # Leading non-Arabic punctuation on any word boundary (e.g. ,اطلب)
    'leading_punct': re.compile('(?:(?<=\\s)|^)[^\\wء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ\\(\\[' + chr(0xAB) + '"' + "'" + "]+(?=[" + chr(0x600) + '-' + chr(0x6FF) + "])"),
    # Full post has no Arabic content at all
    'punct_only':     re.compile(
                        rf'^[^{ARABIC_BLOCK}{ARABIC_SUPP}{ARABIC_EXT_A}{ARABIC_PRES_A}{ARABIC_PRES_B}]*$'),
}


# Keep only Arabic letters + whitespace — strips ALL punctuation and digits
# Replace non-Arabic chars sandwiched between Arabic letters with a space
# so adjacent punctuation (e.g. وأيضاً:كذا) doesn't merge words.
BOUNDARY_FIX = re.compile(r'(?<=[ء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ])[^ء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ\s]+(?=[ء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ])')
KEEP_ONLY_ARABIC = re.compile(r'[^ء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴࢶ-ࢽﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ\s]')

@dataclass
class CleanStats:
    total_raw:             int = 0
    removed_empty:         int = 0
    removed_deleted:       int = 0
    removed_spaced_ar:     int = 0
    removed_punct_only:    int = 0
    removed_low_ar:        int = 0
    removed_short:         int = 0
    removed_numbers:       int = 0
    final_clean:           int = 0
    chars_before:          int = 0
    chars_after:           int = 0
    english_words_removed: int = 0
    urls_removed:          int = 0
    quran_chars_removed:   int = 0
    extra: dict            = field(default_factory=dict)

    def report(self):
        removed = self.total_raw - self.final_clean
        ratio   = self.chars_after / max(self.chars_before, 1)
        print("\n" + "═"*52)
        print("  RedditCleaner — Corpus Stats")
        print("═"*52)
        print(f"  Raw rows              : {self.total_raw:>10,}")
        print(f"  Removed (empty/null)  : {self.removed_empty:>10,}")
        print(f"  Removed ([deleted])   : {self.removed_deleted:>10,}")
        print(f"  Removed (spaced AR)   : {self.removed_spaced_ar:>10,}")
        print(f"  Removed (punct only)  : {self.removed_punct_only:>10,}")
        print(f"  Removed (low Arabic)  : {self.removed_low_ar:>10,}")
        print(f"  Removed (too short)   : {self.removed_short:>10,}")
        print(f"  Removed (numbers only): {self.removed_numbers:>10,}")
        print(f"  ─────────────────────────────────────")
        print(f"  Total removed         : {removed:>10,}  ({removed/max(self.total_raw,1)*100:.1f}%)")
        print(f"  Final corpus size     : {self.final_clean:>10,}  ({self.final_clean/max(self.total_raw,1)*100:.1f}%)")
        print(f"  ─────────────────────────────────────")
        print(f"  Chars before          : {self.chars_before:>10,}")
        print(f"  Chars after           : {self.chars_after:>10,}")
        print(f"  Char retention        : {ratio*100:>9.1f}%")
        print(f"  English words removed : {self.english_words_removed:>10,}")
        print(f"  URLs removed          : {self.urls_removed:>10,}")
        print(f"  Quranic chars stripped: {self.quran_chars_removed:>10,}")
        print("═"*52 + "\n")


class RedditCleaner:
    def __init__(
        self,
        min_arabic_ratio: float = 0.60,
        min_chars:        int   = 20,
        remove_emojis:    bool  = True,
        strip_english:    bool  = True,
    ):
        self.min_arabic_ratio = min_arabic_ratio
        self.min_chars        = min_chars
        self.remove_emojis    = remove_emojis
        self.strip_english    = strip_english
        self.stats            = CleanStats()

    # ── Spaced-Arabic detection (whole-post heuristic) ────────────────────────
    @staticmethod
    def _is_spaced_arabic(text: str) -> bool:
        """True if >50% of tokens are single Arabic letters (Uthmanic/Quranic layout)."""
        tokens = text.split()
        if len(tokens) < 6:
            return False
        single = sum(1 for tok in tokens if _SINGLE_AR.match(tok))
        return (single / len(tokens)) > 0.50

    # ── Strip inline spaced-Arabic spans, keep surrounding normal Arabic ───────
    @staticmethod
    def _strip_spaced_arabic_spans(text: str) -> str:
        """Remove contiguous spans of spaced single Arabic letters (Quranic verse text)
        while preserving normal Arabic words around them."""
        # Match runs of 5+ tokens that are single Arabic chars separated by spaces
        return re.sub(
            rf'(?:(?:[{ARABIC_BLOCK}{ARABIC_SUPP}{ARABIC_EXT_A}] ){{5,}}'
            rf'[{ARABIC_BLOCK}{ARABIC_SUPP}{ARABIC_EXT_A}])',
            ' ', text
        )

    # ── Text-level cleaning ───────────────────────────────────────────────────
    def _clean_text(self, text: str) -> str:
        t = text

        # 1. Strip Quranic special characters
        before = len(t)
        t = QURAN_CHARS.sub('', t)
        self.stats.quran_chars_removed += (before - len(t))

        # 2. Remove URLs, mentions, subreddits, hashtags, HTML
        urls = len(PATTERNS['url'].findall(t))
        self.stats.urls_removed += urls
        t = PATTERNS['url'].sub(' ', t)
        for key in ('mention', 'subreddit', 'hashtag', 'html_entity', 'html_tag'):
            t = PATTERNS[key].sub(' ', t)

        # 3. Emojis
        if self.remove_emojis:
            t = PATTERNS['emoji'].sub(' ', t)

        # 4. Strip English words
        if self.strip_english:
            en_words = PATTERNS['english_word'].findall(t)
            self.stats.english_words_removed += len(en_words)
            t = PATTERNS['english_word'].sub(' ', t)

        # 5. Remove divider lines (__ _ _ _,  ------, ======)
        t = PATTERNS['divider'].sub(' ', t)

        # 6. Remove spaced punctuation runs (. . .  or - - -)
        t = PATTERNS['spaced_punct'].sub(' ', t)

        # 7. Strip inline spaced Quranic spans without dropping the whole post
        t = self._strip_spaced_arabic_spans(t)

        # 8. Strip leading non-Arabic punctuation glued to Arabic words (,اطلب → اطلب)
        t = PATTERNS['leading_punct'].sub('', t)

        # 9. Normalise Arabic
        t = self._normalise_arabic(t)

        # 10. Flatten ALL newlines → single space (one post = one line)
        t = re.sub(r'\s*\n\s*', ' ', t)

        # 11. Strip ALL punctuation — keep only Arabic letters + spaces
        t = BOUNDARY_FIX.sub(' ', t)  # space between words at punct boundaries
        t = KEEP_ONLY_ARABIC.sub(' ', t)
        t = re.sub(r' {2,}', ' ', t)

        return t.strip()

    @staticmethod
    def _normalise_arabic(text: str) -> str:
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ة',      'ه', text)
        text = re.sub(r'ى',      'ي', text)
        text = re.sub(r'[\u064B-\u065F]', '', text)   # diacritics
        text = re.sub(r'ـ', '', text)                  # tatweel
        return text

    # ── Row-level filters ─────────────────────────────────────────────────────
    def _arabic_ratio(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        return sum(1 for w in words if ARABIC_RE.search(w)) / len(words)

    # ── DataFrame entry point ─────────────────────────────────────────────────
    def clean_data_frame(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        s = self.stats
        s.total_raw    = len(df)
        s.chars_before = df[col].astype(str).str.len().sum()

        # 1. Drop nulls / empty
        mask = df[col].notna() & (df[col].astype(str).str.strip() != '')
        s.removed_empty = (~mask).sum()
        df = df[mask].copy()

        # 2. Drop Reddit placeholders
        _deleted = re.compile(r'^\s*\[?(deleted|removed|AutoModerator)\]?\s*$', re.I)
        mask = ~df[col].apply(lambda t: bool(_deleted.match(t)))
        s.removed_deleted = (~mask).sum()
        df = df[mask].copy()

        # 3. Clean text
        df[col] = df[col].astype(str).apply(self._clean_text)

        # 4. Drop fully-spaced Arabic posts (Uthmanic layout, unrecoverable)
        mask = ~df[col].apply(self._is_spaced_arabic)
        s.removed_spaced_ar = (~mask).sum()
        df = df[mask].copy()

        # 5. Drop posts with no Arabic content left
        mask = ~df[col].apply(lambda t: bool(PATTERNS['punct_only'].match(t)))
        s.removed_punct_only = (~mask).sum()
        df = df[mask].copy()

        # 6. Drop low Arabic-ratio posts
        mask = df[col].apply(lambda t: self._arabic_ratio(t) >= self.min_arabic_ratio)
        s.removed_low_ar = (~mask).sum()
        df = df[mask].copy()

        # 7. Drop too-short posts
        mask = df[col].str.len() >= self.min_chars
        s.removed_short = (~mask).sum()
        df = df[mask].copy()

        # 8. Drop number-only posts
        mask = ~df[col].apply(lambda t: bool(PATTERNS['number_only'].match(t.strip())))
        s.removed_numbers = (~mask).sum()
        df = df[mask].copy()

        s.final_clean = len(df)
        s.chars_after = df[col].str.len().sum()
        s.report()

        return df.reset_index(drop=True)