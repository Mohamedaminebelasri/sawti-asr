import numpy as np

from sawti_asr.augment.noise_bank import mix_at_snr
from sawti_asr.data.normalize import has_latin, normalize_text
from sawti_asr.data.prepare import doda_speaker
from sawti_asr.evaluate.metrics import compute_metrics, relative_gain


def test_normalize_arabic_and_latin():
    assert normalize_text("أنَا غادي لـلـ Réunion!") == "انا غادي لل réunion"
    assert normalize_text("  كيداير ؟  ") == "كيداير"


def test_has_latin():
    assert has_latin("غادي نمشي للـ réunion")
    assert not has_latin("كيداير لاباس")


def test_mix_at_snr_reaches_target():
    rng = np.random.default_rng(0)
    speech = (0.1 * np.sin(np.linspace(0, 2000, 16000))).astype(np.float32)
    noise = rng.normal(0, 1, 8000).astype(np.float32)  # plus court : sera répété
    mixed = mix_at_snr(speech, noise, snr_db=10, rng=rng)
    added = mixed - speech
    snr = 10 * np.log10(np.mean(speech ** 2) / np.mean(added ** 2))
    assert abs(snr - 10) < 0.1


def test_metrics_and_gain():
    m = compute_metrics(["غادي نمشي", "labas"], ["غادي نمشي", ""])
    assert m["n"] == 2 and abs(m["wer"] - 1 / 3) < 1e-9
    assert abs(relative_gain(0.5, 0.4) - 0.2) < 1e-9


def test_doda_speaker():
    ranges = {"F1": [[0, 999], [8000, 8999]], "M1": [[1000, 1999]]}
    assert doda_speaker(8500, ranges) == "F1"
    assert doda_speaker(1500, ranges) == "M1"
    assert doda_speaker(5000, ranges) == "unk"
