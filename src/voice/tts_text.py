"""Text normalization for spoken output (TTS frontend cleanup).

Converts display text into something the offline Piper/VITS frontend
can actually pronounce: numbers become words, years are read naturally,
CamelCase executable names are split (``WindowsTerminal`` ->
``Windows Terminal``), and filesystem paths are reduced to a speakable
name. The original display text is never modified — this only affects
what is synthesized.
"""

from __future__ import annotations

import re

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def _int_to_words(n: int) -> str:
    """Convert 0 <= n <= 9999 to English words."""
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + (" " + _ONES[r] if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + _int_to_words(r) if r else "")
    th, r = divmod(n, 1000)
    return _int_to_words(th) + " thousand" + (
        " " + _int_to_words(r) if r else ""
    )


def _year_to_words(n: int) -> str:
    """Read a 4-digit year naturally: 2026 -> 'twenty twenty six'."""
    if 1000 <= n <= 2099:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return _int_to_words(hi) + " hundred"
        return (
            _int_to_words(hi)
            + " "
            + (_ONES[lo] if lo < 20 else _int_to_words(lo))
        )
    return _int_to_words(n)


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_PATHISH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/|%)")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WS_RE = re.compile(r"\s+")


def _split_camel(token: str) -> str:
    return _CAMEL_RE.sub(" ", token)


def _strip_path(token: str) -> str:
    """Reduce a path-like token to a speakable basename."""
    base = re.split(r"[\\/]+", token)[-1]
    base = re.sub(r"\.(exe|lnk|bat|cmd|msi|dll)$", "", base, flags=re.I)
    return base or token


def normalize_for_speech(text: str) -> str:
    """Return a TTS-friendly version of ``text``.

    - Paths/URLs-ish tokens become their basename without extension.
    - CamelCase and snake_case names get word spacing.
    - Digit runs become spoken numbers; 4-digit years (1900-2099)
      are read as years.
    """
    if not text:
        return text
    out_tokens = []
    for raw in text.split():
        token = raw.strip("\"'()[]{}<>")
        punct_pre = raw[: len(raw) - len(raw.lstrip("\"'()[]{}<>"))]
        punct_post = raw[len(raw.rstrip("\"'()[]{}<>")):]
        core = token
        if _PATHISH_RE.search(core):
            core = _strip_path(core)
        core = core.replace("_", " ")
        core = _split_camel(core)
        out_tokens.append(punct_pre + core + punct_post)
    text = " ".join(out_tokens)

    def _num_repl(m: re.Match) -> str:
        s = m.group(0)
        if "." in s:
            # Decimals: read digits individually-ish via int parts.
            whole, frac = s.split(".", 1)
            return (
                _int_to_words(int(whole)) + " point "
                + " ".join(_ONES[int(d)] for d in frac)
            )
        n = int(s)
        if 1900 <= n <= 2099 and len(s) == 4:
            return _year_to_words(n)
        return _int_to_words(n)

    text = _NUM_RE.sub(_num_repl, text)
    return _WS_RE.sub(" ", text).strip()
