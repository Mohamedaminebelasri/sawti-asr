"""Étape 1 : préparer les données (DODa, MoulSot, Casablanca) au format HF.

Exemples :
  python scripts/prepare_data.py
  python scripts/prepare_data.py --no-moulsot          # test rapide avec DODa seul
"""
import argparse
import logging

from sawti_asr.config import load_config, set_seed
from sawti_asr.data.prepare import build_dataset, summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml", "configs/data.yaml"])
    ap.add_argument("--no-moulsot", action="store_true")
    ap.add_argument("--no-casablanca", action="store_true")
    args = ap.parse_args()

    cfg = load_config(*args.configs)
    set_seed(cfg["seed"])
    dd = build_dataset(cfg, use_moulsot=not args.no_moulsot,
                       use_casablanca=not args.no_casablanca)
    print(summary(dd))

    dd.save_to_disk(cfg["paths"]["processed"])
    logging.info("Sauvegardé dans %s", cfg["paths"]["processed"])
    if cfg["hub"]["push"]:
        if not cfg["hub"]["repo_id"]:
            raise ValueError("hub.repo_id manquant dans configs/base.yaml")
        dd.push_to_hub(cfg["hub"]["repo_id"], private=True)  # toujours PRIVÉ (licences)
        logging.info("Envoyé sur le Hub (privé) : %s", cfg["hub"]["repo_id"])


if __name__ == "__main__":
    main()
