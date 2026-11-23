"""Chargement et fusion de fichiers de configuration YAML."""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Charge plusieurs YAML ; les suivants écrasent les précédents."""
    cfg: dict[str, Any] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
