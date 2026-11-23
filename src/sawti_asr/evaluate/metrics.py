"""WER, CER et gain relatif."""
from __future__ import annotations

import jiwer

from ..data.normalize import normalize_text

_EMPTY = "∅"  # remplace une prédiction vide (même nombre d'erreurs pour le WER)


def compute_metrics(references: list[str], predictions: list[str],
                    text_cfg: dict | None = None) -> dict[str, float]:
    text_cfg = text_cfg or {}
    refs, hyps = [], []
    for r, h in zip(references, predictions):
        r = normalize_text(r, **text_cfg)
        if not r:
            continue  # une référence vide n'est pas évaluable
        refs.append(r)
        hyps.append(normalize_text(h, **text_cfg) or _EMPTY)
    if not refs:
        raise ValueError("Aucune référence non vide à évaluer")
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "n": len(refs),
    }


def relative_gain(wer_baseline: float, wer_model: float) -> float:
    """Gain = (WER_baseline - WER_modèle) / WER_baseline."""
    return (wer_baseline - wer_model) / wer_baseline
