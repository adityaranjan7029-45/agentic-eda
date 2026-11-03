"""The server's size modes (deploy/small.env, deploy/large.env) must accept our own demo data.

Found live in step 16: small mode allowed 100 columns, but the sample credit dataset has 122,
so "Try a sample credit dataset" was refused on the free server.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "data" / "samples" / "credit_applications.csv"


def _mode(name: str) -> dict[str, str]:
    lines = (ROOT / "deploy" / f"{name}.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if line.strip() and not line.startswith("#"))


@pytest.mark.parametrize("name", ["small", "large"])
def test_the_sample_dataset_fits_every_server_mode(name):
    from eda_engine.synth.credit import generate_credit_applications

    mode = _mode(name)
    generated = generate_credit_applications(n_rows=3000, seed=1).data   # what POST /runs/sample analyses
    assert generated.shape[1] <= int(mode["MAX_COLUMNS"])
    assert len(generated) <= int(mode["MAX_ROWS"])
    with SAMPLE_CSV.open() as f:                                         # the CSV shipped in the repo
        assert len(f.readline().split(",")) <= int(mode["MAX_COLUMNS"])
    assert SAMPLE_CSV.stat().st_size <= int(mode["MAX_UPLOAD_MB"]) * 1024 * 1024


def test_large_mode_is_never_smaller_than_small_mode():
    small, large = _mode("small"), _mode("large")
    for key in ("WORKERS", "MAX_UPLOAD_MB", "MAX_ROWS", "MAX_COLUMNS"):
        assert int(large[key]) >= int(small[key]), key
