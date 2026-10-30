"""Lossless phase storage policy. No model, data, seed or training setting changes."""
from pathlib import Path
import torch
from .common import atomic_torch_save, file_hash, read_json, source_hash, write_json


def completed_storage_policy(payload, phase, status):
    payload = Path(payload)
    policy = {"version": 2, "packaging_source_hash": source_hash(), "lossless": True,
        "model_quantization": False, "excluded_files": [], "checkpoint_references": {},
        "recovery": "All available recovery states retained for failed phases"}
    if status != "COMPLETE":
        return policy
    if phase == "smoke":
        changes = []
        for checkpoint in sorted((payload / "smoke").glob("*/smoke_checkpoint.pt")):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            removed = [key for key in ("optimizer", "scheduler", "scaler", "rng") if key in saved]
            if not removed:
                continue
            original_sha = file_hash(checkpoint)
            original_bytes = checkpoint.stat().st_size
            compact = {key: value for key, value in saved.items() if key not in removed}
            temporary = checkpoint.with_name("verified_model_only.pt")
            atomic_torch_save(compact, temporary)
            reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
            if reloaded["model"].keys() != saved["model"].keys():
                raise ValueError("Storage rewrite changed model parameter names")
            for name, parameter in saved["model"].items():
                if parameter.dtype != reloaded["model"][name].dtype or not torch.equal(parameter, reloaded["model"][name]):
                    raise ValueError("Storage rewrite altered model weights: " + name)
            if reloaded["config"] != saved["config"] or reloaded["metadata"] != saved["metadata"]:
                raise ValueError("Storage rewrite altered checkpoint configuration")
            del reloaded, compact, saved
            temporary.replace(checkpoint)
            changes.append({"path": checkpoint.relative_to(payload).as_posix(), "original_sha256": original_sha,
                "original_bytes": original_bytes, "compact_bytes": checkpoint.stat().st_size,
                "compact_sha256": file_hash(checkpoint), "removed_completed_smoke_only": removed,
                "every_model_tensor_equal": True, "same_tensor_dtypes": True})
        if changes:
            write_json(payload / "smoke/checkpoint_storage.json", {"changes": changes,
                "note": "Smoke save/reload already passed. Full model weights retained exactly; no continued smoke optimization is required."})
        policy["recovery"] = "Successful smoke retains exact model weights; transient optimizer state omitted"
    if phase not in ("prepare", "smoke", "evaluate", "finalize"):
        # A completed experiment is never continued by later phases. Its exact
        # best model, histories, config and hashes remain; last.pt is only useful
        # while an experiment is unfinished. Live last.pt is still written normally.
        policy["excluded_files"] = [p.relative_to(payload).as_posix() for p in (payload / "runs").glob("*/last.pt")]
        policy["recovery"] = "Successful run retains exact best model; unfinished/failed run retains last.pt with full optimizer/RNG state"
    if phase == "finalize":
        policy["submission_layout"] = "single_flat_archive"
        policy["excluded_files"].append("question_2_submission.zip")
    if phase == "evaluate":
        # All selected checkpoints are already retained by their training phase.
        # Finalization needs results/records, not duplicate copies of these weights.
        for checkpoint in (payload / "runs").glob("*/best.pt"):
            completion = read_json(checkpoint.parent / "complete.json")
            checksum = file_hash(checkpoint)
            if checksum != completion["best_sha256"]:
                raise ValueError("Evaluation checkpoint differs from verified training evidence")
            relative = checkpoint.relative_to(payload).as_posix()
            policy["excluded_files"].append(relative)
            policy["checkpoint_references"][relative] = {"sha256": checksum,
                "retained_in": f"Q2_{checkpoint.parent.name}_results.zip"}
        policy["recovery"] = "Evaluation preserves predictions/results and exact hashes referencing the retained training archives"
    return policy
