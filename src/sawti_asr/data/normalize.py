"""Normalisation du texte : Darija en alphabet arabe, français en latin.

La même fonction est appliquée aux références et aux prédictions avant le
calcul du WER/CER, pour que l'évaluation soit cohérente.
"""
from __future__ import annotations

import re
import unicodedata

_AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_ALEF_VARIANTS = re.compile(r"[آأإٱ]")  # آ أ إ ٱ
_TATWEEL = "ـ"
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
_GAF_VARIANTS = re.compile(r"[ڭگ]")  # ڭ گ -> ݣ (U+0763)
_GAF = "ݣ"
_SPACES = re.compile(r"\s+")


def _strip_punct_and_symbols(text: str) -> str:
    return "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text
    )


def normalize_text(
    text: str,
    remove_diacritics: bool = True,
    normalize_alef: bool = True,
    remove_tatweel: bool = True,
    remove_punctuation: bool = True,
    lowercase_latin: bool = True,
    unify_gaf: bool = True,
) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    if remove_diacritics:
        text = _AR_DIACRITICS.sub("", text)
    if normalize_alef:
        text = _ALEF_VARIANTS.sub("ا", text)
    if unify_gaf:
        text = _GAF_VARIANTS.sub(_GAF, text)  # le son « g » s'écrit de trois façons
    if remove_tatweel:
        text = text.replace(_TATWEEL, "")
    if remove_punctuation:
        text = _strip_punct_and_symbols(text)
    if lowercase_latin:
        text = text.lower()  # l'arabe n'a pas de casse : seul le latin change
    return _SPACES.sub(" ", text).strip()


def has_latin(text: str) -> bool:
    """Vrai si le texte contient des lettres latines (indice de code-switching)."""
    return bool(_LATIN.search(text or ""))


def has_arabic(text: str) -> bool:
    """Vrai si le texte contient des caractères arabes (U+0600–U+06FF, U+0750–U+077F)."""
    return bool(_ARABIC.search(text or ""))


def is_code_switching(text: str) -> bool:
    """Vrai si le texte mélange l'alphabet arabe et l'alphabet latin."""
    return has_arabic(text) and has_latin(text)
