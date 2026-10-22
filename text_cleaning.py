# text_cleaning.py
import re
from typing import Tuple


REPEATED_CHAR_RE = re.compile(r'(.)\1{6,}')
SPACED_LETTER_RE = re.compile(r'\b(?:[A-Za-z]\s+){2,}[A-Za-z0-9]\b')
NON_WORD_RE = re.compile(r'[^A-Za-z0-9\u4e00-\u9fff\s\-\_\.\,\:\;/\(\)%]+')
WORD_RE = re.compile(r'[A-Za-z][A-Za-z0-9\-_/\.]{1,}')


def normalize_basic_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # 修复类似 i t e m 1 / C P U / N V L i n k
    def _merge_spaced_letters(match):
        s = match.group(0)
        merged = s.replace(" ", "")
        return merged

    text = SPACED_LETTER_RE.sub(_merge_spaced_letters, text)

    # 去掉长重复字符
    text = REPEATED_CHAR_RE.sub(lambda m: m.group(1) * 2, text)

    return text.strip()


def compute_dirty_score(text: str) -> dict:
    if not text:
        return {
            "length": 0,
            "repeated_char_runs": 0,
            "spaced_letter_patterns": 0,
            "alnum_ratio": 0.0,
            "word_count": 0,
            "readable_word_ratio": 0.0,
            "non_word_noise_ratio": 0.0,
            "is_dirty": True,
        }

    repeated_char_runs = len(REPEATED_CHAR_RE.findall(text))
    spaced_letter_patterns = len(SPACED_LETTER_RE.findall(text))
    alnum_chars = sum(ch.isalnum() for ch in text)
    alnum_ratio = alnum_chars / max(len(text), 1)

    words = WORD_RE.findall(text)
    word_count = len(words)
    readable_word_ratio = word_count / max(len(text.split()), 1)

    noisy_chars = len(NON_WORD_RE.findall(text))
    non_word_noise_ratio = noisy_chars / max(len(text), 1)

    is_dirty = (
        repeated_char_runs >= 3
        or spaced_letter_patterns >= 2
        or alnum_ratio < 0.35
        or (len(text) > 200 and readable_word_ratio < 0.2)
        or non_word_noise_ratio > 0.2
    )

    return {
        "length": len(text),
        "repeated_char_runs": repeated_char_runs,
        "spaced_letter_patterns": spaced_letter_patterns,
        "alnum_ratio": alnum_ratio,
        "word_count": word_count,
        "readable_word_ratio": readable_word_ratio,
        "non_word_noise_ratio": non_word_noise_ratio,
        "is_dirty": is_dirty,
    }


def clean_extracted_text(text: str) -> Tuple[str, dict]:
    cleaned = normalize_basic_text(text)
    stats = compute_dirty_score(cleaned)
    return cleaned, stats