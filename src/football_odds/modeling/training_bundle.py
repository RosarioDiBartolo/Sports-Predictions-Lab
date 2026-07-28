"""Portable, checksum-verified inputs for remote model training."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_training_bundle(
    project: Path,
    preflight_path: Path,
    destination: Path,
) -> Path:
    """Copy exactly the inputs accepted by a successful immutable preflight."""
    report: dict[str, Any] = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError("Only a successful preflight can create a training bundle.")
    if destination.exists():
        raise FileExistsError(f"Bundle destination already exists: {destination}")

    destination.mkdir(parents=True)
    bundled_inputs = []
    try:
        for recorded in report["inputs"]:
            source = Path(recorded["path"]).resolve()
            try:
                relative = source.relative_to(project.resolve())
            except ValueError as error:
                raise ValueError(f"Input is outside the project: {source}") from error
            actual_sha256 = _sha256(source)
            if actual_sha256 != recorded["sha256"]:
                raise ValueError(f"Input changed after preflight: {source}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            bundled_inputs.append(
                {
                    "path": relative.as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": actual_sha256,
                }
            )
        manifest = {
            "source_run_id": report["run_id"],
            "dataset_version": report["dataset_version"],
            "inputs": bundled_inputs,
        }
        manifest_path = destination / "training_bundle.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path
    except BaseException:
        shutil.rmtree(destination)
        raise


def verify_training_bundle(bundle: Path) -> dict[str, Any]:
    """Fail when a bundled input differs from its recorded bytes or checksum."""
    manifest_path = bundle / "training_bundle.manifest.json"
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    for recorded in manifest["inputs"]:
        path = bundle / recorded["path"]
        if not path.is_file():
            raise ValueError(f"Missing bundled input: {recorded['path']}")
        if path.stat().st_size != recorded["bytes"]:
            raise ValueError(f"Wrong byte size: {recorded['path']}")
        if _sha256(path) != recorded["sha256"]:
            raise ValueError(f"Checksum mismatch: {recorded['path']}")
    return manifest
