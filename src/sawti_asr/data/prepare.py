"""Préparation des données au format Hugging Face Datasets.

Sorties (DatasetDict) :
  train, validation         : DODa + sous-ensemble de MoulSot
  test_clean_doda           : DODa, locuteurs jamais vus à l'entraînement
  test_moulsot              : test officiel MoulSot (chaînes exclues du train)
  test_cs_moulsot           : test officiel MoulSot, extraits arabe + latin (CS)
  test_casablanca_all       : Casablanca-Maroc complet (validation + test), test externe

Colonnes communes : audio, text, source, speaker, duration.
"""
from __future__ import annotations

import logging
import zlib
from typing import Callable, Iterable, Iterator

import numpy as np
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    Value,
    get_dataset_split_names,
    load_dataset,
)

from .normalize import has_arabic, is_code_switching, normalize_text

log = logging.getLogger(__name__)

TEXT_CANDIDATES = ("transcription", "text", "sentence", "transcript", "darija")


def features(sr: int) -> Features:
    return Features(
        {
            "audio": Audio(sampling_rate=sr),
            "text": Value("string"),
            "source": Value("string"),
            "speaker": Value("string"),
            "duration": Value("float32"),
        }
    )


def _check_split(repo: str, config: str | None, split: str) -> None:
    names = get_dataset_split_names(repo, config)
    if split not in names:
        raise ValueError(
            f"Split '{split}' absent de {repo} ({config}). Splits disponibles : {names}. "
            "Corrigez configs/data.yaml."
        )


def _detect_text_column(columns: Iterable[str], wanted: str | None) -> str:
    columns = list(columns)
    if wanted:
        if wanted not in columns:
            raise ValueError(f"Colonne '{wanted}' absente. Colonnes : {columns}")
        return wanted
    for c in TEXT_CANDIDATES:
        if c in columns:
            log.info("Colonne de texte détectée : %s", c)
            return c
    raise ValueError(f"Impossible de trouver la colonne de texte parmi {columns}")


def _row(example: dict, text: str, source: str, speaker: str, sr: int, cfg: dict):
    """Construit une ligne propre, ou None si elle doit être filtrée."""
    audio = example["audio"]
    array = np.asarray(audio["array"], dtype=np.float32)
    if array.ndim > 1:  # sécurité : passage en mono
        array = array.mean(axis=0)
    duration = len(array) / sr
    if not (cfg["min_duration_s"] <= duration <= cfg["max_duration_s"]):
        return None
    if not text or not np.isfinite(array).all() or np.abs(array).max() == 0:
        return None
    return {
        "audio": {"array": array, "sampling_rate": sr},
        "text": text,
        "source": source,
        "speaker": speaker,
        "duration": float(duration),
    }


def _norm(text: str, cfg: dict) -> str:
    return normalize_text(text, **cfg["text"])


def _from_generator(gen: Callable[..., Iterator[dict]], sr: int, **kwargs) -> Dataset:
    # from_generator écrit sur disque au fur et à mesure : pas d'explosion de RAM.
    # Les kwargs (str/bool uniquement) distinguent les générateurs dans le cache.
    return Dataset.from_generator(gen, features=features(sr), gen_kwargs=kwargs or None)


# ---------------------------------------------------------------- DODa
def doda_speaker(index: int, ranges: dict[str, list[list[int]]]) -> str:
    for spk, spans in ranges.items():
        for start, end in spans:
            if start <= index <= end:
                return spk
    return "unk"


def prepare_doda(cfg: dict) -> dict[str, Dataset]:
    c, sr = cfg["doda"], cfg["sampling_rate"]
    ds = load_dataset(c["repo"], split=c["split"])
    text_col = _detect_text_column(ds.column_names, c["text_column"])
    ds = ds.cast_column("audio", Audio(sampling_rate=sr))
    ranges = c.get("speaker_ranges") or {}

    if ranges:
        test_spk, val_spk = set(c["test_speakers"]), set(c["val_speakers"])
        if not test_spk:
            raise ValueError("Renseignez doda.test_speakers dans configs/data.yaml")

        def split_of(i: int) -> str:
            spk = doda_speaker(i, ranges)
            if spk in test_spk:
                return "test"
            return "validation" if spk in val_spk else "train"
    else:
        log.warning(
            "doda.speaker_ranges est vide : split ALÉATOIRE (pas disjoint par "
            "locuteur). À corriger avant les vraies expériences."
        )
        rng = np.random.default_rng(cfg["seed"])
        draws = rng.random(len(ds))
        t, v = c["test_ratio_fallback"], c["val_ratio_fallback"]

        def split_of(i: int) -> str:
            return "test" if draws[i] < t else ("validation" if draws[i] < t + v else "train")

    out = {}
    for name in ("train", "validation", "test"):

        def gen(name):
            for i, ex in enumerate(ds):
                if split_of(i) != name:
                    continue
                row = _row(ex, _norm(ex[text_col], cfg), "doda",
                           doda_speaker(i, ranges), sr, cfg)
                if row:
                    yield row

        out[name] = _from_generator(gen, sr, name=name)
        log.info("DODa %s : %d exemples", name, len(out[name]))
    return out


# ---------------------------------------------------------------- MoulSot
def prepare_moulsot(cfg: dict) -> dict[str, Dataset]:
    c, sr = cfg["moulsot"], cfg["sampling_rate"]
    ch_col = c["channel_column"]

    # 1) Test officiel (petit) : on récupère ses chaînes pour les exclure du train.
    _check_split(c["repo"], c["test_config"], c["test_split"])
    test = load_dataset(c["repo"], c["test_config"], split=c["test_split"])
    text_col = _detect_text_column(test.column_names, c["text_column"])
    test = test.cast_column("audio", Audio(sampling_rate=sr))
    test_channels = set(test[ch_col]) if ch_col in test.column_names else set()
    log.info("MoulSot : %d chaînes dans le test officiel", len(test_channels))

    def gen_test(cs_only: bool):
        for ex in test:
            raw = ex[text_col]
            # Le filtre CS porte sur le texte brut (transcriptions Gemini).
            if cs_only and not is_code_switching(raw):
                continue
            row = _row(ex, _norm(raw, cfg), "moulsot",
                       str(ex.get(ch_col, "unk")), sr, cfg)
            if row:
                yield row

    # 2) Train en streaming (évite de télécharger tout le dataset).
    def is_val(channel: str) -> bool:
        return zlib.crc32(channel.encode()) % c["val_channel_modulo"] == 0

    def gen_stream(want_val: bool):
        budget_s = c["max_train_hours"] * 3600 * (0.05 if want_val else 1.0)
        stream = load_dataset(c["repo"], c["train_config"], split=c["train_split"],
                              streaming=True)
        stream = stream.cast_column("audio", Audio(sampling_rate=sr))
        total = 0.0
        for ex in stream:
            channel = str(ex.get(ch_col, "unk"))
            if channel in test_channels or is_val(channel) != want_val:
                continue
            row = _row(ex, _norm(ex.get(text_col, ""), cfg), "moulsot", channel, sr, cfg)
            if row:
                total += row["duration"]
                yield row
                if total >= budget_s:
                    break

    out = {
        "test": _from_generator(gen_test, sr, cs_only=False),
        "test_cs": _from_generator(gen_test, sr, cs_only=True),
        "train": _from_generator(gen_stream, sr, want_val=False),
        "validation": _from_generator(gen_stream, sr, want_val=True),
    }
    for k, v in out.items():
        log.info("MoulSot %s : %d exemples, %.1f h", k, len(v), sum(v["duration"]) / 3600)
    return out


# ---------------------------------------------------------------- Casablanca
def prepare_casablanca(cfg: dict) -> dict[str, Dataset]:
    c, sr = cfg["casablanca"], cfg["sampling_rate"]
    parts = []
    for split in c["splits"]:
        _check_split(c["repo"], c["config"], split)
        parts.append(load_dataset(c["repo"], c["config"], split=split))
    text_col = _detect_text_column(parts[0].column_names, c["text_column"])

    # Pas de jeu CS ici : le français y est translittéré en arabe (pas de latin).
    def gen():
        for part in parts:
            part = part.cast_column("audio", Audio(sampling_rate=sr))
            for ex in part:
                row = _row(ex, _norm(ex[text_col], cfg), "casablanca", "unk", sr, cfg)
                if row:
                    yield row

    all_ds = _from_generator(gen, sr)
    log.info("Casablanca : %d segments", len(all_ds))
    return {"all": all_ds}


# ---------------------------------------------------------------- Assemblage
def build_dataset(cfg: dict, use_moulsot: bool = True, use_casablanca: bool = True) -> DatasetDict:
    from datasets import concatenate_datasets

    doda = prepare_doda(cfg)
    train, val = [doda["train"]], [doda["validation"]]
    dd = {"test_clean_doda": doda["test"]}

    if use_moulsot:
        ms = prepare_moulsot(cfg)
        train.append(ms["train"])
        val.append(ms["validation"])
        dd["test_moulsot"] = ms["test"]
        dd["test_cs_moulsot"] = ms["test_cs"]
    if use_casablanca:
        cb = prepare_casablanca(cfg)
        dd["test_casablanca_all"] = cb["all"]

    train_ds = concatenate_datasets(train)
    val_ds = concatenate_datasets(val)
    if cfg.get("drop_latin_only_train", False):
        # Retire de train/validation les extraits sans aucune lettre arabe
        # (Arabizi ou langue étrangère). Les jeux de test ne sont pas touchés.
        train_ds = _drop_latin_only(train_ds, "train")
        val_ds = _drop_latin_only(val_ds, "validation")
    dd["train"] = train_ds.shuffle(seed=cfg["seed"])
    dd["validation"] = val_ds
    return DatasetDict(dd)


def _drop_latin_only(ds: Dataset, name: str) -> Dataset:
    # input_columns="text" : seule la colonne texte est lue, l'audio n'est pas décodé.
    kept = ds.filter(has_arabic, input_columns="text")
    log.info("%s : %d extraits sans lettre arabe retirés (%d -> %d)",
             name, len(ds) - len(kept), len(ds), len(kept))
    return kept


def summary(dd: DatasetDict) -> str:
    lines = ["split | exemples | heures"]
    for name, ds in dd.items():
        lines.append(f"{name} | {len(ds)} | {sum(ds['duration']) / 3600:.2f}")
    return "\n".join(lines)
