"""Banque de bruits ESC-50 et mélange parole + bruit à un SNR donné.

Les folds 1-4 servent à l'augmentation (entraînement), le fold de test (5 par
défaut) sert uniquement à construire le jeu de test bruité : aucune fuite.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NoiseClip:
    array: np.ndarray
    category: str
    filename: str


def load_esc50(esc50_dir: str | Path, folds: list[int], sr: int = 16000,
               min_rms: float = 1e-4) -> list[NoiseClip]:
    import librosa
    import pandas as pd

    root = Path(esc50_dir)
    meta = pd.read_csv(root / "meta" / "esc50.csv")
    meta = meta[meta["fold"].isin(folds)]
    clips = []
    for _, r in meta.iterrows():
        y, _ = librosa.load(root / "audio" / r["filename"], sr=sr, mono=True)
        if np.sqrt(np.mean(y ** 2)) >= min_rms:  # écarte les clips quasi silencieux
            clips.append(NoiseClip(y.astype(np.float32), r["category"], r["filename"]))
    if not clips:
        raise RuntimeError(f"Aucun bruit chargé depuis {root} pour les folds {folds}")
    return clips


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator) -> np.ndarray:
    """Ajoute `noise` à `speech` pour obtenir un SNR de `snr_db` décibels."""
    speech = speech.astype(np.float32)
    n = len(speech)
    if len(noise) < n:  # on répète le bruit s'il est plus court que la phrase
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    start = rng.integers(0, len(noise) - n + 1)
    noise = noise[start:start + n].astype(np.float32)

    p_speech = np.mean(speech ** 2) + 1e-10
    p_noise = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    mixed = speech + scale * noise
    peak = np.abs(mixed).max()
    if peak > 1.0:  # évite la saturation sans changer le SNR
        mixed = mixed / peak
    return mixed.astype(np.float32)
