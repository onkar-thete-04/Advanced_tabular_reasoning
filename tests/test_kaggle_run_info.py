"""Test for the Kaggle run_info.json writer."""

import json


def test_write_run_info(tmp_path):
    from kaggle.train_kaggle import write_run_info

    path = write_run_info(str(tmp_path), {"model": "x", "sft_seconds": 1.5})
    data = json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))
    assert data["model"] == "x"
    assert data["sft_seconds"] == 1.5
