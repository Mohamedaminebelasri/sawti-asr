"""Début de session : recharge data/processed depuis le Hub, puis les jeux bruités.

À lancer au début de chaque session Kaggle/Colab, avant les expériences :
  python scripts/fetch_data.py
1) data/processed absent -> rechargé depuis le Hub (dataset privé hub.repo_id).
2) data/processed présent et data/test_noisy absent -> scripts/build_test_sets.py
   (graine fixe : les jeux bruités sont identiques d'une session à l'autre).
Ne fait rien de plus si le dataset n'est pas (encore) sur le Hub.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

from sawti_asr.config import load_config
from sawti_asr.data.hub import ensure_processed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml", "configs/data.yaml"])
    ap.add_argument("--no-noisy", action="store_true", help="ne pas reconstruire les jeux bruités")
    args = ap.parse_args()
    cfg = load_config(*args.configs)

    status = ensure_processed(cfg)
    print("data/processed :", status)
    if status == "absent" or args.no_noisy:
        return
    noisy = Path(cfg["paths"]["noisy_test"])
    if noisy.exists():
        print("data/test_noisy : local")
        return
    script = Path(__file__).with_name("build_test_sets.py")
    subprocess.run([sys.executable, str(script), "--configs", *args.configs], check=True)
    print("data/test_noisy : reconstruit")


if __name__ == "__main__":
    main()
