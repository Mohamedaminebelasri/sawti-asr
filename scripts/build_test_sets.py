"""Étape 2 : construire les jeux de test BRUITÉS, une seule fois (graine fixe).

Chaque phrase du test propre est mélangée à un bruit ESC-50 du fold de test,
à chaque SNR (20/10/5/0 dB). Toutes les expériences réutilisent ces fichiers.
"""
import argparse
import logging

import numpy as np
from datasets import Audio, Dataset, DatasetDict, Features, Value, load_from_disk

from sawti_asr.augment.noise_bank import load_esc50, mix_at_snr
from sawti_asr.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml", "configs/data.yaml"])
    args = ap.parse_args()
    cfg = load_config(*args.configs)
    sr, nc = cfg["sampling_rate"], cfg["noise"]

    clean = load_from_disk(cfg["paths"]["processed"])[nc["noisy_source"]]
    noises = load_esc50(nc["esc50_dir"], folds=[nc["test_fold"]], sr=sr, min_rms=nc["min_rms"])
    logging.info("%d phrases propres, %d bruits de test", len(clean), len(noises))

    feats = Features({
        "audio": Audio(sampling_rate=sr), "text": Value("string"),
        "snr_db": Value("float32"), "noise_category": Value("string"),
        "noise_file": Value("string"),
    })

    def gen(snr):
        rng = np.random.default_rng(cfg["seed"] + int(snr))
        for ex in clean:
            noise = noises[rng.integers(len(noises))]
            mixed = mix_at_snr(np.asarray(ex["audio"]["array"]), noise.array, snr, rng)
            yield {"audio": {"array": mixed, "sampling_rate": sr}, "text": ex["text"],
                   "snr_db": float(snr), "noise_category": noise.category,
                   "noise_file": noise.filename}

    dd = DatasetDict({
        f"snr_{int(s)}": Dataset.from_generator(gen, features=feats, gen_kwargs={"snr": s})
        for s in nc["snrs_db"]
    })
    dd.save_to_disk(cfg["paths"]["noisy_test"])
    logging.info("Jeux bruités sauvegardés dans %s : %s", cfg["paths"]["noisy_test"], list(dd))


if __name__ == "__main__":
    main()
