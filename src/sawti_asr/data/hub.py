"""Sauvegarde et rechargement des données préparées via le Hugging Face Hub.

Le dataset du Hub est toujours PRIVÉ (licences : Casablanca est CC BY-NC-ND).
Le jeton est lu dans la variable d'environnement HF_TOKEN, jamais dans le code.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def push_processed(dd, repo_id: str) -> None:
    """Envoie le DatasetDict préparé sur le Hub, en dataset privé."""
    if not repo_id:
        raise ValueError("hub.repo_id manquant dans configs/base.yaml")
    dd.push_to_hub(repo_id, private=True)
    log.info("Envoyé sur le Hub (privé) : %s", repo_id)


def ensure_processed(cfg: dict) -> str:
    """Garantit que data/processed existe, en le rechargeant depuis le Hub si besoin.

    Renvoie "local" (déjà là), "hub" (rechargé) ou "absent" (rien à recharger).
    """
    processed = Path(cfg["paths"]["processed"])
    if processed.exists():
        log.info("%s existe déjà : rien à recharger", processed)
        return "local"
    repo_id = (cfg.get("hub") or {}).get("repo_id")
    if not repo_id:
        log.warning("hub.repo_id vide : impossible de recharger %s", processed)
        return "absent"

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        HfApi().dataset_info(repo_id)
    except RepositoryNotFoundError:
        log.warning("Dataset %s introuvable sur le Hub (pas encore envoyé, ou jeton "
                    "sans accès) : rien à recharger", repo_id)
        return "absent"
    dd = load_dataset(repo_id)
    dd.save_to_disk(str(processed))
    log.info("Rechargé depuis le Hub : %s -> %s (%s)", repo_id, processed,
             ", ".join(f"{k}={len(v)}" for k, v in dd.items()))
    return "hub"
