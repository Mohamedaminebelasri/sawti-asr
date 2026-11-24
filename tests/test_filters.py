"""Filtre Arabizi : seul test_moulsot est filtré parmi les jeux de test."""
import pytest

datasets = pytest.importorskip("datasets")

from sawti_asr.data.prepare import _drop_latin_only_test  # noqa: E402

TEXTES = ["كيداير لاباس", "kidayr labas", "غادي نمشي للـ réunion", "salut cava"]


def _jeux():
    ds = datasets.Dataset.from_dict({"text": TEXTES})
    return {"test_moulsot": ds, "test_clean_doda": ds, "test_casablanca_all": ds}


def test_drop_latin_only_test_filtre_seulement_test_moulsot():
    dd = _jeux()
    _drop_latin_only_test(dd, {"drop_latin_only_test": True})
    assert dd["test_moulsot"]["text"] == ["كيداير لاباس", "غادي نمشي للـ réunion"]
    assert len(dd["test_clean_doda"]) == 4
    assert len(dd["test_casablanca_all"]) == 4


def test_drop_latin_only_test_desactive():
    dd = _jeux()
    _drop_latin_only_test(dd, {"drop_latin_only_test": False})
    assert len(dd["test_moulsot"]) == 4
    _drop_latin_only_test(dd, {})
    assert len(dd["test_moulsot"]) == 4
