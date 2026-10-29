"""Token budgeting without a tokenizer dependency.

The usual rule of thumb, "1 token ≈ 4 characters", is calibrated on English
prose. Compact tabular text (pipes, digits, snake_case names) packs far fewer
characters per token: measured with the o200k tokenizer family that the
gpt-oss models use, chars/4 undercounts our profiles by ~45%, which would
silently blow through Groq's rate limits.

So instead of counting characters, `estimate_tokens` mimics how the tokenizer
splits text: runs of letters (long ones cost more), digits in groups of up to
three, each punctuation mark, and long runs of whitespace (padded tables like
`df.to_string()` spend real tokens on spaces). Against exact o200k counts on profiles of
five different datasets this lands at 1.12-1.24x the true count with the 5%
margin below, so it errs on the safe side without wasting much budget.
Callers can pass an exact counter instead.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

TokenCounter = Callable[[str], int]

_PIECE_RE = re.compile(r"[A-Za-z]+|\d{1,3}|[^\sA-Za-z\d]|\s{2,}")
CHARS_PER_WORD_TOKEN = 4
CHARS_PER_SPACE_TOKEN = 8
SAFETY_MARGIN = 1.05


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for compact, data-heavy text."""
    pieces = 0
    for match in _PIECE_RE.finditer(text):
        piece = match.group()
        if piece[0].isalpha():
            pieces += math.ceil(len(piece) / CHARS_PER_WORD_TOKEN)
        elif piece[0].isspace():
            pieces += math.ceil(len(piece) / CHARS_PER_SPACE_TOKEN)
        else:
            pieces += 1
    return math.ceil(pieces * SAFETY_MARGIN)
