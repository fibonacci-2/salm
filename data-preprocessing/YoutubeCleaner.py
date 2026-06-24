"""YoutubeCleaner - Arabic YouTube comments cleaner for LLM pretraining."""
import re, pandas as pd
from dataclasses import dataclass, field

_AR    = '\u0600-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-\ufeff'
_FOR   = 'A-zÀ-ɏЀ-ӿऀ-ॿঀ-\u09ff一-鿿\u0590-\u05ffͰ-Ͽ가-\ud7af\u3040-ヿ'
_QC    = 'ٱٰؕۖ-ۭࣔ-ࣿ﮲-﯁\u0600-\u0605ؐ-ؚ\u06dd-۟۠-۪ۤۧۨ-ۭ'
_BIDI  = '\u200b-\u200f\u202a-\u202e\u2066-\u206f\ufeff'
_DIACR = 'ً-ٟ'
_EMO   = '😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰🤀-🧿☀-⛿🨀-\U0001faff⌀-⏿'
_HEARTS= '❤♥'

ARABIC_RE         = re.compile('[' + _AR   + ']')
FOREIGN_SCRIPT_RE = re.compile('[' + _FOR  + ']')
QURAN_CHARS       = re.compile('[' + _QC   + ']')
BIDI_CONTROLS     = re.compile('[' + _BIDI + ']')

_EMO_PAT  = '[' + _EMO + ']+'
_ONLY_EMO = '^[' + '\\s' + _HEARTS + _EMO + ']+$'

YT_PAT = {
    'url'          : re.compile(r'https?://\S+|www\.\S+', re.I),
    'mention'      : re.compile(r'@\w+'),
    'hashtag'      : re.compile(r'#\w+'),
    'html_entity'  : re.compile(r'&\s*#?\s*[a-zA-Z0-9][\sa-zA-Z0-9]*\s*;', re.I),
    'html_tag'     : re.compile(r'<[^>]+>'),
    'timestamp'    : re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b'),
    'emoji'        : re.compile(_EMO_PAT, re.UNICODE),
    'emoji_only'   : re.compile(_ONLY_EMO),
    'divider'      : re.compile(r'(?:[-_=*~#][ \t]*){3,}'),
    'spaced_punct' : re.compile('(?:[^\\w\\s\u0600-ۿݐ-ݿࢠ-ࣿ][ \\t]*){3,}'),
    'repeated_char': re.compile(r'(.)\1{3,}'),
    'repeated_punct': re.compile('([^\\w\\s\u0600-ۿݐ-ݿࢠ-ࣿ])\\1{2,}'),
    'newline'      : re.compile(r'\s*\n\s*'),
    'extra_space'  : re.compile(r'[ \t]{2,}'),
    'english_word' : re.compile(r'\b[A-Za-z][A-Za-z0-9\'\-]*\b'),
    'leading_punct': re.compile('(?:(?<=\\s)|^)[^\\w' + _AR + '\\(\\[' + chr(0xAB) + '"' + "'" + "]+(?=[" + chr(0x600) + '-' + chr(0x6FF) + "])"),
    'number_only'  : re.compile('[^\\d\\s.,\\-+\\(\\)' + _DIACR + ']'),
    'no_arabic'    : re.compile('^[^' + _AR + ']*$'),
}

_AR_NORM = [
    (re.compile('[إأآا]'), 'ا'),
    (re.compile('ة'), 'ه'),
    (re.compile('ى'), 'ي'),
    (re.compile('[' + _DIACR + ']'), ''),
    (re.compile('ـ'), ''),
]

# Keep only Arabic letters + whitespace — strips ALL punctuation and digits
_KEEP_ONLY_ARABIC = r'[^ء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴࢶ-ࢽﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ\s]'
KEEP_ONLY_ARABIC = re.compile(_KEEP_ONLY_ARABIC)
BOUNDARY_FIX = re.compile(r'(?<=[ء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ])[^ء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ\s]+(?=[ء-غف-يٮ-ۓەۮ-ۯۺ-ۼۿݐ-ݿࢠ-ࢴﭐ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼ])')

@dataclass
class YTCleanStats:
    total_raw: int = 0
    removed_empty: int = 0
    removed_emoji_only: int = 0
    removed_foreign: int = 0
    removed_no_arabic: int = 0
    removed_low_ar: int = 0
    removed_short: int = 0
    removed_numbers: int = 0
    final_clean: int = 0
    chars_before: int = 0
    chars_after: int = 0
    english_words_removed: int = 0
    urls_removed: int = 0
    timestamps_removed: int = 0
    quran_chars_removed: int = 0
    extra: dict = field(default_factory=dict)
    def report(self):
        removed = self.total_raw - self.final_clean
        ratio   = self.chars_after / max(self.chars_before, 1)
        w = 54
        print("\n" + "="*w)
        print("  YoutubeCleaner - Corpus Stats")
        print("="*w)
        for label, val in [
            ("Raw rows",               self.total_raw),
            ("Removed (empty/null)",   self.removed_empty),
            ("Removed (emoji only)",   self.removed_emoji_only),
            ("Removed (foreign lang)", self.removed_foreign),
            ("Removed (no Arabic)",    self.removed_no_arabic),
            ("Removed (low Arabic)",   self.removed_low_ar),
            ("Removed (too short)",    self.removed_short),
            ("Removed (numbers only)", self.removed_numbers),
        ]:
            print(f"  {label:<26}: {val:>8,}")
        print("  " + "-"*(w-2))
        print(f"  Total removed             : {removed:>8,}  ({removed/max(self.total_raw,1)*100:.1f}%)")
        print(f"  Final corpus size         : {self.final_clean:>8,}  ({self.final_clean/max(self.total_raw,1)*100:.1f}%)")
        print("  " + "-"*(w-2))
        print(f"  Chars before              : {self.chars_before:>8,}")
        print(f"  Chars after               : {self.chars_after:>8,}")
        print(f"  Char retention            : {ratio*100:>7.1f}%")
        print(f"  English words removed     : {self.english_words_removed:>8,}")
        print(f"  URLs removed              : {self.urls_removed:>8,}")
        print(f"  Timestamps removed        : {self.timestamps_removed:>8,}")
        print(f"  Quranic chars stripped    : {self.quran_chars_removed:>8,}")
        print("="*w + "\n")

class YoutubeCleaner:
    def __init__(self, min_arabic_ratio=0.60, min_chars=10,
                 remove_emojis=True, strip_english=True, max_foreign_ratio=0.20):
        self.min_arabic_ratio  = min_arabic_ratio
        self.min_chars         = min_chars
        self.remove_emojis     = remove_emojis
        self.strip_english     = strip_english
        self.max_foreign_ratio = max_foreign_ratio
        self.stats             = YTCleanStats()

    def _foreign_ratio(self, text):
        tokens = text.split()
        if not tokens: return 0.0
        return sum(1 for t in tokens if FOREIGN_SCRIPT_RE.search(t) and not ARABIC_RE.search(t)) / len(tokens)

    def _arabic_ratio(self, text):
        tokens = text.split()
        if not tokens: return 0.0
        return sum(1 for t in tokens if ARABIC_RE.search(t)) / len(tokens)

    def _clean_text(self, text):
        t = BIDI_CONTROLS.sub("", text)
        before = len(t)
        t = QURAN_CHARS.sub("", t)
        self.stats.quran_chars_removed += before - len(t)
        self.stats.urls_removed += len(YT_PAT['url'].findall(t))
        t = YT_PAT['url'].sub(' ', t)
        self.stats.timestamps_removed += len(YT_PAT['timestamp'].findall(t))
        t = YT_PAT['timestamp'].sub(' ', t)
        for k in ('mention','hashtag','html_entity','html_tag'):
            t = YT_PAT[k].sub(' ', t)
        if self.remove_emojis:
            t = YT_PAT['emoji'].sub(' ', t)
        if self.strip_english:
            en = YT_PAT['english_word'].findall(t)
            self.stats.english_words_removed += len(en)
            t = YT_PAT['english_word'].sub(' ', t)
        for k in ('divider','spaced_punct'):
            t = YT_PAT[k].sub(' ', t)
        t = YT_PAT['leading_punct'].sub('', t)
        for pat, repl in _AR_NORM:
            t = pat.sub(repl, t)
        t = YT_PAT['newline'].sub(' ', t)

        # Strip ALL punctuation — keep only Arabic letters + spaces
        t = BOUNDARY_FIX.sub(' ', t)  # space between words at punct boundaries
        t = KEEP_ONLY_ARABIC.sub(' ', t)
        t = re.sub(r' {2,}', ' ', t)
        return t.strip()

    def clean_data_frame(self, df, col):
        s = self.stats
        df = df.copy()
        df[col] = df[col].astype(object).where(df[col].notna(), "").astype(str)
        s.total_raw    = len(df)
        s.chars_before = df[col].str.len().sum()
        mask = df[col].apply(lambda t: t.strip() != '')
        s.removed_empty = (~mask).sum(); df = df[mask].copy()
        mask = ~df[col].apply(lambda t: bool(YT_PAT['emoji_only'].match(t.strip())))
        s.removed_emoji_only = (~mask).sum(); df = df[mask].copy()
        mask = df[col].apply(lambda t: self._foreign_ratio(t) <= self.max_foreign_ratio)
        s.removed_foreign = (~mask).sum(); df = df[mask].copy()
        df[col] = df[col].apply(self._clean_text)
        mask = ~df[col].apply(lambda t: bool(YT_PAT['no_arabic'].match(t)))
        s.removed_no_arabic = (~mask).sum(); df = df[mask].copy()
        mask = df[col].apply(lambda t: self._arabic_ratio(t) >= self.min_arabic_ratio)
        s.removed_low_ar = (~mask).sum(); df = df[mask].copy()
        mask = df[col].apply(lambda t: len(t) >= self.min_chars)
        s.removed_short = (~mask).sum(); df = df[mask].copy()
        mask = df[col].apply(lambda t: bool(YT_PAT['number_only'].search(t.strip())))
        s.removed_numbers = (~mask).sum(); df = df[mask].copy()
        s.final_clean = len(df)
        s.chars_after = df[col].str.len().sum()
        s.report()
        return df.reset_index(drop=True)