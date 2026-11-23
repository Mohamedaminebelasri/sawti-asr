"""Connexion à MLflow (serveur DagsHub) via des variables d'environnement.

À définir avant de lancer un script (par ex. dans les « Secrets » Kaggle/Colab) :
  MLFLOW_TRACKING_URI      = https://dagshub.com/<utilisateur>/<depot>.mlflow
  MLFLOW_TRACKING_USERNAME = <utilisateur DagsHub>
  MLFLOW_TRACKING_PASSWORD = <jeton d'accès DagsHub>
Sans ces variables, MLflow enregistre en local (./mlruns), perdu à la fin de session.
"""
from __future__ import annotations

import logging
import os

import mlflow

log = logging.getLogger(__name__)


def setup_mlflow(experiment: str) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        log.info("MLflow -> %s", uri)
    else:
        log.warning("MLFLOW_TRACKING_URI absent : enregistrement LOCAL dans ./mlruns")
    mlflow.set_experiment(experiment)


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Aplatit une config imbriquée pour mlflow.log_params."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = str(v)[:250]  # limite de taille d'un paramètre MLflow
    return out
