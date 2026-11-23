# Sawti-ASR-FineTuner & Data-Augmenter

Pipeline ASR & MLOps pour la Darija marocaine et le code-switching français–arabe :
augmentation audio, fine-tuning de Whisper (full FT et LoRA), évaluation sur 3 jeux
de test (propre, bruité, code-switching), suivi et registre de modèles avec MLflow.

Projet d'initiation 4A, filière IA, ENSAM Meknès. Encadrants : Pr. Hosni, Pr. Menaoui.

## État : semaine 1

| Étape | Script | Sortie |
|---|---|---|
| 1. Préparer les données | `scripts/prepare_data.py` | `data/processed/` |
| 2. Construire les tests bruités | `scripts/build_test_sets.py` | `data/test_noisy/` |
| 3. E0 Whisper zero-shot | `scripts/run_e0.py` | `results/e0/` + run MLflow |

## Données

| Source | Rôle | Licence |
|---|---|---|
| atlasia/DODa-audio-dataset | entraînement + test propre (split par locuteur) | MIT |
| atlasia/MoulSot-Full | entraînement (≈20 h) + test spontané (split par chaîne) + test code-switching (extraits arabe + latin, transcriptions Gemini) | non déclarée |
| UBC-NLP/Casablanca (Morocco) | test externe uniquement (le français y est translittéré en arabe) | CC BY-NC-ND 4.0 |
| ESC-50 | bruits : folds 1–4 entraînement, fold 5 test | CC BY-NC 3.0 |

Convention d'écriture : Darija en alphabet arabe, mots français en latin.
**Aucun audio n'est versionné dans ce dépôt.**

## Installation (Kaggle ou Colab)

```bash
git clone https://github.com/<user>/sawti-asr.git && cd sawti-asr
pip install -r requirements.txt && pip install -e .
git clone https://github.com/karolpiczak/ESC-50 data/ESC-50
huggingface-cli login     # accepter d'abord les conditions de DODa et Casablanca sur le Hub
```

Variables d'environnement pour MLflow (serveur DagsHub, à mettre dans les secrets) :

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/sawti-asr.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<jeton DagsHub>
```

## Lancer la semaine 1

```bash
pytest -q                                              # tests rapides
python scripts/prepare_data.py --no-moulsot            # 1er essai avec DODa seul
python scripts/prepare_data.py                         # version complète
python scripts/build_test_sets.py
python scripts/run_e0.py --model openai/whisper-tiny --max-samples 50   # essai
python scripts/run_e0.py --model openai/whisper-base                    # E0 complet
```

## À faire avant les vraies expériences

- Remplir `doda.speaker_ranges`, `test_speakers` et `val_speakers` dans
  `configs/data.yaml` à partir de la carte de DODa (sinon le split est aléatoire).
- Vérifier le nom du split de test de MoulSot (`moulsot.test_split`).
- Vérifier que Casablanca contient bien du latin (jeu code-switching).
- Figer les versions : `pip freeze > requirements-lock.txt`.
