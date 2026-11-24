from sawti_asr.data.hub import ensure_processed


def test_ensure_processed_local(tmp_path):
    (tmp_path / "processed").mkdir()
    cfg = {"paths": {"processed": str(tmp_path / "processed")}, "hub": {"repo_id": "x/y"}}
    assert ensure_processed(cfg) == "local"


def test_ensure_processed_without_repo(tmp_path):
    cfg = {"paths": {"processed": str(tmp_path / "absent")}, "hub": {"repo_id": None}}
    assert ensure_processed(cfg) == "absent"
