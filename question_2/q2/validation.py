"""Evidence checks used before skipping completed training or consuming prerequisites."""
import math
from pathlib import Path
from .common import digest, file_hash, read_json


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def verify_completed_run(directory, expected_signature=None):
    directory = Path(directory)
    complete = read_json(directory / "complete.json")
    run = read_json(directory / "run.json")
    history = read_json(directory / "history.json")
    if complete["status"] != "complete" or complete["signature"] != run["signature"]:
        raise ValueError(f"Incomplete/mismatched training: {directory}")
    if expected_signature and complete["signature"] != expected_signature:
        raise ValueError("Completed run configuration/data/source signature changed")
    if not complete.get("evidence_sha256"):
        raise ValueError("Completion record lacks evidence hashes; rerun with the current implementation")
    for name, checksum in complete["evidence_sha256"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError(f"Corrupt or replaced training evidence: {name}")
    if file_hash(directory / "best.pt") != complete["best_sha256"]:
        raise ValueError("Corrupt or replaced checkpoint")
    if len(history) != complete["epochs_completed"] or [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("History is incomplete or epoch numbering is inconsistent")
    if not history or not finite_tree(history):
        raise ValueError("Missing or nonfinite training history")
    expected = run["config"]["dataset"]["expected_sizes"]
    for row in history:
        for split in ("train", "validation"):
            if row[split]["n"] != expected[split]:
                raise ValueError("Training history has the wrong sample count")
    best = next((r for r in history if r["epoch"] == complete["best_epoch"]), None)
    if best is None or not math.isclose(best["validation"]["macro_f1"], complete["best_validation_macro_f1"], abs_tol=1e-10):
        raise ValueError("Selected epoch and recorded validation score disagree")
    return {"run": run, "complete": complete, "history": history}
