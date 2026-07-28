import hashlib
import json

import pytest

from football_odds.modeling.training_bundle import (
    create_training_bundle,
    verify_training_bundle,
)


def test_bundle_copies_and_verifies_preflight_inputs(tmp_path):
    project = tmp_path / "project"
    source = project / "data/processed/input.csv"
    source.parent.mkdir(parents=True)
    source.write_text("match_id\nm1\n", encoding="utf-8")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    preflight = project / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "run_id": "accepted-run",
                "passed": True,
                "dataset_version": checksum,
                "inputs": [
                    {
                        "path": str(source),
                        "bytes": source.stat().st_size,
                        "sha256": checksum,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    destination = tmp_path / "bundle"
    manifest_path = create_training_bundle(project, preflight, destination)
    manifest = verify_training_bundle(destination)

    assert manifest_path.is_file()
    assert manifest["source_run_id"] == "accepted-run"
    bundled_input = destination / "data/processed/input.csv"
    assert bundled_input.read_bytes() == source.read_bytes()


def test_bundle_rejects_inputs_changed_after_preflight(tmp_path):
    project = tmp_path / "project"
    source = project / "input.csv"
    project.mkdir()
    source.write_text("changed", encoding="utf-8")
    preflight = project / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "run_id": "run",
                "passed": True,
                "dataset_version": "old",
                "inputs": [
                    {
                        "path": str(source),
                        "bytes": 3,
                        "sha256": hashlib.sha256(b"old").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed after preflight"):
        create_training_bundle(project, preflight, tmp_path / "bundle")

    assert not (tmp_path / "bundle").exists()
