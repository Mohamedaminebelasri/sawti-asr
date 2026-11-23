"""Étape 3 : E0 — Whisper zero-shot sur tous les jeux de test, suivi dans MLflow.

Exemples :
  python scripts/run_e0.py
  python scripts/run_e0.py --model openai/whisper-tiny --max-samples 50   # test rapide
"""
import argparse
import logging
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from sawti_asr.config import load_config, set_seed
from sawti_asr.evaluate.metrics import compute_metrics
from sawti_asr.tracking.mlflow_utils import flatten, setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def transcribe(ds, model, processor, cfg, device, dtype):
    preds, bs = [], cfg["eval"]["batch_size"]
    for i in range(0, len(ds), bs):
        batch = ds[i:i + bs]
        arrays = [a["array"] for a in batch["audio"]]
        feats = processor.feature_extractor(
            arrays, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype)
        with torch.no_grad():
            ids = model.generate(
                feats, language=cfg["model"]["language"], task=cfg["model"]["task"],
                max_new_tokens=cfg["eval"]["max_new_tokens"],
            )
        preds += processor.batch_decode(ids, skip_special_tokens=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=[
        "configs/base.yaml", "configs/data.yaml", "configs/experiments/e0_zeroshot.yaml"])
    ap.add_argument("--model", help="remplace model.name")
    ap.add_argument("--max-samples", type=int, help="remplace eval.max_samples")
    args = ap.parse_args()

    cfg = load_config(*args.configs)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.max_samples:
        cfg["eval"]["max_samples"] = args.max_samples
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    name = cfg["model"]["name"]
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(device)
    model.eval()

    # Jeux de test : propres/CS depuis data/processed, bruités depuis data/test_noisy
    test_sets = {}
    processed = load_from_disk(cfg["paths"]["processed"])
    for split in cfg["eval"]["test_sets"]:
        if split in processed:
            test_sets[split] = processed[split]
        else:
            logging.warning("Jeu absent, ignoré : %s", split)
    noisy_path = Path(cfg["paths"]["noisy_test"])
    if cfg["eval"]["include_noisy"] and noisy_path.exists():
        for split, ds in load_from_disk(str(noisy_path)).items():
            test_sets[f"test_noisy_{split}"] = ds

    setup_mlflow(cfg["mlflow"]["experiment"])
    out_dir = Path(cfg["paths"]["results"]) / "e0"
    out_dir.mkdir(parents=True, exist_ok=True)
    short = name.split("/")[-1]

    with mlflow.start_run(run_name=f"E0_{short}"):
        mlflow.set_tags({"experiment_id": "E0", "model": name, "device": device})
        mlflow.log_params(flatten(cfg))
        rows = []
        for split, ds in test_sets.items():
            n = cfg["eval"]["max_samples"]
            if n:
                ds = ds.select(range(min(n, len(ds))))
            t0 = time.time()
            preds = transcribe(ds, model, processor, cfg, device, dtype)
            m = compute_metrics(ds["text"], preds, cfg["text"])
            m["seconds"] = time.time() - t0
            logging.info("%s : WER=%.3f CER=%.3f (n=%d)", split, m["wer"], m["cer"], m["n"])
            mlflow.log_metrics({f"{split}/wer": m["wer"], f"{split}/cer": m["cer"],
                                f"{split}/seconds": m["seconds"]})
            rows.append({"split": split, **m})
            pred_file = out_dir / f"{short}_{split}_predictions.csv"
            pd.DataFrame({"reference": ds["text"], "prediction": preds}).to_csv(
                pred_file, index=False)
            mlflow.log_artifact(str(pred_file), artifact_path="predictions")

        table = out_dir / f"{short}_metrics.csv"
        pd.DataFrame(rows).to_csv(table, index=False)
        mlflow.log_artifact(str(table))
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
