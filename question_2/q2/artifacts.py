"""Streamed archive inspection, integrity checks and safe prerequisite extraction."""
from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
import numpy as np
import pandas as pd
from .common import digest, file_hash, read_json, write_json
from .metrics import classification_metrics
from .validation import finite_tree

TRAIN_PHASES = [f"{model}_seed{seed}" for model in ("baseline", "roberta_mean", "roberta_first") for seed in (42, 123)]


class ArtifactReader:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(path) if self.path.is_file() else None
        self.prefix = ""
        if self.archive:
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Archive contains duplicate entries")
            for info in self.archive.infolist():
                relative = PurePosixPath(info.filename)
                if relative.is_absolute() or ".." in relative.parts or "\\" in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Unsafe archive entry")
            manifests = [n for n in names if n.endswith("/phase_manifest.json") or n == "phase_manifest.json"]
            if len(manifests) != 1:
                raise ValueError("Archive must contain exactly one phase_manifest.json")
            self.prefix = manifests[0][:-len("phase_manifest.json")]
            if any(not n.startswith(self.prefix) for n in names):
                raise ValueError("Archive has files outside its declared phase root")
            self.names = {n[len(self.prefix):] for n in names if not n.endswith("/")}
        else:
            if not (self.path / "phase_manifest.json").exists():
                raise ValueError(f"Not a phase archive/directory: {self.path}")
            self.names = {p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()
                          and ".cache" not in p.parts and "test_cache" not in p.parts and not p.name.endswith(".tmp")}
            excluded = read_json(self.path / "phase_manifest.json").get("storage_policy", {}).get("excluded_files", [])
            self.names.difference_update(excluded)
            if any(p.is_symlink() for p in self.path.rglob("*")):
                raise ValueError("Artifact directories must not contain symlinks")

    @contextmanager
    def open(self, name):
        if name not in self.names:
            raise ValueError(f"Missing phase artifact: {name}")
        handle = self.archive.open(self.prefix + name) if self.archive else open(self.path / name, "rb")
        try:
            yield handle
        finally:
            handle.close()

    def json(self, name):
        with self.open(name) as handle:
            return json.load(handle)

    def csv(self, name):
        with self.open(name) as handle:
            return pd.read_csv(handle)

    def size(self, name):
        return self.archive.getinfo(self.prefix + name).file_size if self.archive else (self.path / name).stat().st_size

    def checksum(self, name):
        result = hashlib.sha256()
        with self.open(name) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()

    def close(self):
        if self.archive:
            self.archive.close()


def require(reader, names):
    missing = sorted(set(names) - reader.names)
    if missing:
        raise ValueError("Missing required outputs: " + ", ".join(missing))


def check_predictions(reader, path, saved_metrics, expected_n, reference=None):
    frame = reader.csv(path)
    columns = [f"probability_{i}" for i in range(1, 6)]
    if len(frame) != expected_n or frame.id.duplicated().any() or frame.id.isna().any():
        raise ValueError(f"Wrong prediction count or duplicate/missing IDs: {path}")
    if not frame.true_stars.isin(range(1, 6)).all() or not frame.predicted_stars.isin(range(1, 6)).all():
        raise ValueError("Predictions contain invalid star labels")
    probability = frame[columns].to_numpy()
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    if not np.allclose(probability.sum(1), 1, atol=1e-5):
        raise ValueError("Class probabilities do not sum to one")
    if not np.array_equal(probability.argmax(1) + 1, frame.predicted_stars.to_numpy()):
        raise ValueError("Predicted labels disagree with probability argmax")
    if reference is not None and (frame.id.tolist() != reference.id.tolist() or frame.true_stars.tolist() != reference.true_stars.tolist()):
        raise ValueError("Prediction IDs/labels differ from the frozen sample index")
    actual = classification_metrics(frame.true_stars.to_numpy() - 1, frame.predicted_stars.to_numpy() - 1)
    if not finite_tree(saved_metrics):
        raise ValueError("Nonfinite saved evaluation metrics")
    for key, value in actual.items():
        if not np.allclose(np.asarray(saved_metrics[key]), np.asarray(value), atol=1e-8):
            raise ValueError(f"Saved metrics disagree with predictions: {path}: {key}")
    return frame


def check_training(reader, name, phase_manifest, require_recovery=True):
    prefix = f"runs/{name}/"
    required = ["run.json", "complete.json", "history.json", "environment-freeze.txt", "training.log"]
    reference = phase_manifest.get("storage_policy", {}).get("checkpoint_references", {}).get(prefix + "best.pt")
    if reference is None:
        required.append("best.pt")
    require(reader, [prefix + n for n in required])
    if require_recovery:
        require(reader, [prefix + "last.pt"])
    run, complete, history = (reader.json(prefix + n) for n in ("run.json", "complete.json", "history.json"))
    if complete["status"] != "complete" or complete["signature"] != run["signature"]:
        raise ValueError("Training is not complete")
    from .compatibility import compatible_prerequisite
    run_source_ok = run["source_hash"] == phase_manifest["source_hash"] or compatible_prerequisite(
        {"phase": name, "source_hash": run["source_hash"]}, phase_manifest["source_hash"])
    if run["config"] != reader.json("config.json") or not run_source_ok:
        raise ValueError("Mixed training configurations or source versions")
    if run["prepared_manifest"] != phase_manifest["prepared_manifest_digest"]:
        raise ValueError("Training used a different prepared dataset")
    if f"{run['model']}_seed{run['seed']}" != name:
        raise ValueError("Experiment model/seed mapping mismatch")
    if 'distributed' in run:
        from .distributed import topology
        expected_topology = topology(run['config'])
        if any(run['distributed'].get(key) != value for key, value in expected_topology.items()) or complete.get('distributed') != expected_topology:
            raise ValueError('Distributed execution changed the frozen global batch/topology')
        require(reader, [prefix + 'distributed_execution.json'])
        if reader.json(prefix + 'distributed_execution.json') != run['distributed']:
            raise ValueError('Distributed execution evidence differs from run metadata')
        devices = run['distributed'].get('devices', [])
        if len(devices) != 2 or {item['rank'] for item in devices} != {0, 1} or len({item['pid'] for item in devices}) != 2:
            raise ValueError('Missing two-worker execution evidence')
        if any(item.get('all_reduce_checksum') != 3 for item in devices):
            raise ValueError('Two-worker communication was not verified')
        if run['distributed'].get('gradient_probe', {}).get('status') != 'PASS':
            raise ValueError('Distributed gradient-equivalence check was not passed')
        if run['distributed'].get('backend') == 'nccl':
            if {item['device'] for item in devices} != {'cuda:0', 'cuda:1'} or any(value <= 0 for value in complete.get('per_rank_peak_gpu_memory_gib', [])) or len(complete.get('per_rank_peak_gpu_memory_gib', [])) != 2:
                raise ValueError('Both CUDA GPUs were not verified active')
        elif run['distributed'].get('backend') != 'gloo' or any(item['device'] != 'cpu' for item in devices):
            raise ValueError('Unexpected distributed backend/device evidence')
        batches = math.ceil(run['config']['dataset']['expected_sizes']['train'] / expected_topology['global_microbatch'])
        updates = math.ceil(batches / expected_topology['accumulation_steps']) * run['config']['roberta']['epochs']
        if run.get('planned_optimizer_updates') != updates:
            raise ValueError('Distributed scheduler update count differs from original experiment')
    evidence = complete.get("evidence_sha256", {})
    if not {"run.json", "history.json", "best.pt", "environment-freeze.txt"}.issubset(evidence):
        raise ValueError("Completed run lacks required evidence hashes")
    for filename, checksum in evidence.items():
        actual = phase_manifest["files"].get(prefix + filename, {}).get("sha256")
        if filename == "best.pt" and reference is not None:
            actual = reference["sha256"]
        if actual != checksum:
            raise ValueError("Training completion evidence changed: " + filename)
    best_sha = reference["sha256"] if reference else phase_manifest["files"][prefix + "best.pt"]["sha256"]
    if best_sha != complete["best_sha256"]:
        raise ValueError("Best checkpoint does not match completion record")
    if len(history) != complete["epochs_completed"] or [r["epoch"] for r in history] != list(range(1, len(history)+1)) or not finite_tree(history):
        raise ValueError("Invalid/incomplete training history")
    for row in history:
        for split in ("train", "validation"):
            if row[split]["n"] != run["config"]["dataset"]["expected_sizes"][split]:
                raise ValueError("Training sample counts do not match the official splits")
    selected = next((r for r in history if r["epoch"] == complete["best_epoch"]), None)
    if selected is None or not np.isclose(selected["validation"]["macro_f1"], complete["best_validation_macro_f1"]):
        raise ValueError("Best epoch and validation result disagree")
    return run


def inspect_artifact(path, expected_phase=None, expected_config=None, expected_source=None, allow_incomplete=False):
    reader = ArtifactReader(path)
    try:
        manifest = reader.json("phase_manifest.json")
        status = manifest["status"]
        if status != reader.json("phase_status.json")["status"] or status not in ("COMPLETE", "FAILED"):
            raise ValueError("Inconsistent phase status")
        if status != "COMPLETE" and not allow_incomplete:
            raise ValueError("Phase has not completed; failed artifacts are recovery-only")
        phase = manifest["phase"]
        if expected_phase and phase != expected_phase:
            raise ValueError(f"Expected phase {expected_phase}, found {phase}")
        config = reader.json("config.json")
        if digest(config) != manifest["config_sha256"] or (expected_config and config != expected_config):
            raise ValueError("Phase configuration mismatch")
        compatibility_used = False
        if expected_source and manifest["source_hash"] != expected_source:
            from .compatibility import compatible_prerequisite
            compatibility_used = compatible_prerequisite(manifest, expected_source)
            if not compatibility_used:
                raise ValueError("Phase source fingerprint differs from the current code")
            print("VERIFIED_LEGACY_PREREQUISITE: known source, unchanged data/model/single-GPU trainer; preflight logging-only update pinned; original provenance preserved", flush=True)
        actual_names = reader.names - {"phase_manifest.json"}
        if actual_names != set(manifest["files"]):
            raise ValueError("Archive inventory is incomplete or has extra files")
        for index, (name, metadata) in enumerate(manifest["files"].items(), 1):
            if reader.size(name) != metadata["bytes"] or reader.checksum(name) != metadata["sha256"]:
                raise ValueError("Artifact hash mismatch: " + name)
            if index % 30 == 0:
                print(f"[inspect] verified {index}/{len(manifest['files'])} files", flush=True)
        if status == "COMPLETE":
            require(reader, ["environment/environment.json", "environment/environment-freeze.txt"])
        if status == "FAILED":
            receipt = {"status": "RECOVERY_ONLY", "phase": phase, "source_hash": manifest["source_hash"],
                       "config_sha256": manifest["config_sha256"], "files_verified": len(manifest["files"])}
            print("RECOVERY_ARTIFACT_VERIFIED; NOT A COMPLETED PREREQUISITE", receipt, flush=True)
            return receipt
        if phase == "prepare":
            require(reader, ["prepared/manifest.json", "prepared/audit/audit.json", "prepared/audit/split_manifest.csv",
                "prepared/audit/split_policy.json", "prepared/data_statistics.json", "prepared/embeddings.npy",
                "prepared/pretrained/model.safetensors", "prepared/vocab.json"])
            audit = reader.json("prepared/audit/audit.json")
            split = reader.csv("prepared/audit/split_manifest.csv")
            if audit["status"] != "PASS" or split.id.duplicated().any() or set(split.split.unique()) != {"train", "validation"}:
                raise ValueError("Dataset audit failed or split IDs overlap")
            if reader.checksum("prepared/audit/split_manifest.csv") != manifest["split_fingerprint"] or audit["split_fingerprint"] != manifest["split_fingerprint"]:
                raise ValueError("Split fingerprint mismatch")
            for name in ("train", "validation"):
                portion = split[split.split == name]
                if len(portion) != config["dataset"]["expected_sizes"][name] or not portion.label.isin(range(5)).all():
                    raise ValueError("Incorrect official split counts/labels")
                require(reader, [f"prepared/{name}.parquet", f"prepared/baseline_{name}.npz",
                                 f"prepared/roberta_{name}/state.json"])
            prepared = reader.json("prepared/manifest.json")
            if prepared["test_accessed"] or any(n.startswith("prepared/test") for n in reader.names):
                raise ValueError("Preparation unexpectedly includes the reserved test split")
            if digest(prepared) != manifest["prepared_manifest_digest"]:
                raise ValueError("Prepared manifest fingerprint mismatch")
            for name, checksum in prepared["files"].items():
                if manifest["files"].get("prepared/" + name, {}).get("sha256") != checksum:
                    raise ValueError("Prepared-file inventory mismatch: " + name)
            stats = reader.json("prepared/data_statistics.json")
            if not finite_tree(stats) or stats["embeddings"]["vocabulary_coverage"] < .25:
                raise ValueError("Invalid preprocessing/embedding statistics")
        elif phase == "smoke":
            summary = reader.json("smoke/SMOKE_COMPLETE.json")
            if summary["status"] != "COMPLETE" or summary["test_accessed"] or {r["model"] for r in summary["models"]} != {"baseline", "roberta_mean"}:
                raise ValueError("Smoke test did not complete both required models")
            reference = None
            for model in summary["models"]:
                prefix = "smoke/" + model["model"] + "/"
                require(reader, [prefix + "smoke_checkpoint.pt", prefix + "metrics.json", prefix + "predictions.csv"])
                metrics = reader.json(prefix + "metrics.json")
                if not metrics["checkpoint_reload_predictions_match"] or model["batches"] < 2:
                    raise ValueError("Smoke forward/backward/reload checks incomplete")
                reference = check_predictions(reader, prefix + "predictions.csv", metrics["metrics"], model["validation_samples"], reference)
        elif phase in TRAIN_PHASES:
            if manifest["experiments"] != [phase]:
                raise ValueError("Wrong number of training experiments in phase")
            check_training(reader, phase, manifest, require_recovery=manifest.get("format", 1) < 2)
        elif phase in ("evaluate", "finalize"):
            expected = [f"{model}_seed{seed}" for model in ("baseline", "roberta_mean", "roberta_first") for seed in config["seeds"]]
            if manifest["experiments"] != expected:
                raise ValueError("Final evidence must cover all six experiments")
            if phase == "evaluate":
                for name in expected:
                    check_training(reader, name, manifest, require_recovery=False)
            require(reader, ["final/EVALUATION_COMPLETE.json", "final/FROZEN_PROTOCOL.json", "final/test_ids.csv",
                "final/test_summary.csv", "final/per_class_summary.csv", "final/validation_summary.csv", "final/ablation_paired.csv",
                "final/REPORT_DRAFT.md", "final/qualitative_examples.csv", "final/token_importance.csv",
                "final/figures/model_comparison.png", "final/figures/pooling_ablation.png", "final/figures/roberta_token_importance.png"])
            reference = reader.csv("final/test_ids.csv")
            protocol = reader.json("final/FROZEN_PROTOCOL.json")
            if len(reference) != config["dataset"]["expected_sizes"]["test"] or reference.id.duplicated().any():
                raise ValueError("Invalid final test sample index")
            for model in ("baseline", "roberta_mean"):
                for seed in config["seeds"]:
                    name = f"{model}_seed{seed}"
                    metric = reader.json(f"final/metrics/{name}.json")
                    if metric["protocol_hash"] != digest(protocol):
                        raise ValueError("Evaluation protocol changed")
                    path = f"final/predictions/{name}.csv"
                    if reader.checksum(path) != metric["prediction_sha256"]:
                        raise ValueError("Prediction file differs from metric record")
                    check_predictions(reader, path, metric["metrics"], len(reference), reference)
            # Recompute aggregate tables, not only the per-seed metric JSON files.
            summary = reader.csv("final/test_summary.csv")
            per_class = reader.csv("final/per_class_summary.csv")
            for model in ("baseline", "roberta_mean"):
                metrics = [reader.json(f"final/metrics/{model}_seed{seed}.json")["metrics"] for seed in config["seeds"]]
                aggregate_row = summary[summary.model == model]
                if len(aggregate_row) != 1:
                    raise ValueError("Missing/duplicate aggregate model row")
                for metric in ("accuracy", "macro_f1", "mae", "adjacent_error_rate", "distant_error_rate"):
                    values = [m[metric] for m in metrics]
                    if not np.isclose(aggregate_row.iloc[0][metric + "_mean"], np.mean(values)) or not np.isclose(aggregate_row.iloc[0][metric + "_std"], np.std(values, ddof=1)):
                        raise ValueError("Aggregate table differs from saved per-seed results")
                for star in range(1, 6):
                    row = per_class[(per_class.model == model) & (per_class.stars == star)]
                    values = [m["per_class_f1"][star - 1] for m in metrics]
                    if len(row) != 1 or not np.allclose([row.iloc[0]["mean"], row.iloc[0]["std"]], [np.mean(values), np.std(values, ddof=1)]):
                        raise ValueError("Per-class aggregate differs from predictions")
            validation_per_seed = reader.csv("final/validation_per_seed.csv")
            validation_summary = reader.csv("final/validation_summary.csv")
            paired = reader.csv("final/ablation_paired.csv")
            if len(validation_per_seed) != len(expected) or len(paired) != len(config["seeds"]):
                raise ValueError("Wrong validation or paired-ablation experiment count")
            selected_metrics = {}
            for name in expected:
                prefix = f"runs/{name}/"
                complete, history = reader.json(prefix + "complete.json"), reader.json(prefix + "history.json")
                best = next(row["validation"] for row in history if row["epoch"] == complete["best_epoch"])
                model, seed = name.rsplit("_seed", 1)
                selected_metrics[(model, int(seed))] = best
                recorded = validation_per_seed[(validation_per_seed.model == model) & (validation_per_seed.seed == int(seed))]
                if len(recorded) != 1 or not np.allclose([recorded.iloc[0][m] for m in ("accuracy", "macro_f1", "mae")], [best[m] for m in ("accuracy", "macro_f1", "mae")]):
                    raise ValueError("Validation table differs from the selected checkpoint epoch")
            for model in ("baseline", "roberta_mean", "roberta_first"):
                row = validation_summary[validation_summary.model == model]
                if len(row) != 1:
                    raise ValueError("Missing validation aggregate model")
                for metric in ("macro_f1", "mae"):
                    values = [selected_metrics[(model, seed)][metric] for seed in config["seeds"]]
                    if not np.allclose([row.iloc[0][metric + "_mean"], row.iloc[0][metric + "_std"]], [np.mean(values), np.std(values, ddof=1)]):
                        raise ValueError("Validation aggregate differs from per-seed measurements")
            for seed in config["seeds"]:
                row = paired[paired.seed == seed]
                if len(row) != 1:
                    raise ValueError("Missing paired ablation seed")
                for metric in ("macro_f1", "mae"):
                    difference = selected_metrics[("roberta_mean", seed)][metric] - selected_metrics[("roberta_first", seed)][metric]
                    if not np.isclose(row.iloc[0][metric + "_mean_minus_first"], difference):
                        raise ValueError("Ablation difference is inconsistent with paired validation results")
            for name in reader.names:
                if name.endswith(".png"):
                    with reader.open(name) as handle:
                        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                            raise ValueError("Unreadable PNG figure: " + name)
            if phase == "finalize" and manifest.get("storage_policy", {}).get("submission_layout") == "single_flat_archive":
                require(reader, ["FINAL_PACKAGE_CHECKED.json", "question_2/README.md", "question_2/config.json",
                                 "question_2/results/test_summary.csv", "question_2/results/REPORT_DRAFT.md"])
                checked = reader.json("FINAL_PACKAGE_CHECKED.json")
                if checked["status"] != "PASS" or reader.json("question_2/config.json") != config:
                    raise ValueError("Flat final package configuration/check differs")
                for name, checksum in checked["submission_files_sha256"].items():
                    if manifest["files"].get(name, {}).get("sha256") != checksum:
                        raise ValueError("Flat submission differs from extracted verified files: " + name)
                if any(n.startswith("question_2/") and ("/test_cache/" in n or n.endswith((".pt", ".safetensors"))) for n in reader.names):
                    raise ValueError("Final submission contains unnecessary data/checkpoints")
            elif phase == "finalize":
                require(reader, ["question_2_submission.zip", "FINAL_PACKAGE_CHECKED.json"])
                checked = reader.json("FINAL_PACKAGE_CHECKED.json")
                if checked["status"] != "PASS" or checked["zip_sha256"] != reader.checksum("question_2_submission.zip"):
                    raise ValueError("Final package verification record differs from ZIP")
                with reader.open("question_2_submission.zip") as handle:
                    data = handle.read()
                with zipfile.ZipFile(io.BytesIO(data)) as package:
                    if package.testzip() is not None:
                        raise ValueError("Nested final ZIP integrity failure")
                    if json.loads(package.read("question_2/config.json")) != config:
                        raise ValueError("Nested submission uses a different configuration")
                    required = {"question_2/README.md", "question_2/config.json", "question_2/results/test_summary.csv", "question_2/results/REPORT_DRAFT.md"}
                    if not required.issubset(package.namelist()):
                        raise ValueError("Nested final ZIP is missing submission files")
                    if any("/test_cache/" in n or "/.venv/" in n or n.endswith((".pt", ".safetensors")) for n in package.namelist()):
                        raise ValueError("Submission contains unnecessary caches/weights")
        else:
            raise ValueError("Unknown phase: " + phase)
        receipt = {"status": "PASS", "phase": phase, "config_sha256": manifest["config_sha256"],
            "source_hash": manifest["source_hash"], "split_fingerprint": manifest["split_fingerprint"],
            "prepared_manifest_digest": manifest["prepared_manifest_digest"], "files_verified": len(manifest["files"]),
            "experiments": manifest["experiments"], "legacy_prerequisite_compatible": compatibility_used, "performance_target": "No numerical target in rubric; completion is not a claim of model quality"}
        print("INSPECTION_PASSED", receipt, flush=True)
        return receipt
    finally:
        reader.close()


def restore_artifact(path, destination, prefixes=None, exclude_basenames=()):
    """Call only after inspection. Extract requested known artifacts, not a shell command."""
    reader = ArtifactReader(path)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for name in sorted(reader.names):
            if PurePosixPath(name).name in exclude_basenames:
                continue
            if name == "phase_manifest.json" or name == "phase_status.json" or name == "config.json" or name.startswith("environment/"):
                continue
            if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
                continue
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Extraction escapes output directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            with reader.open(name) as source, open(target, "wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    finally:
        reader.close()


def package_phase(payload, output, phase, config, prepared_manifest_digest, split_fingerprint, source_hash, experiments, status="COMPLETE"):
    payload, output = Path(payload), Path(output)
    write_json(payload / "phase_status.json", {"phase": phase, "status": status,
               "meaning": "Execution evidence complete; performance/report separate" if status == "COMPLETE" else "Failed execution; recovery-only"})
    from .storage import completed_storage_policy
    storage = completed_storage_policy(payload, phase, status)
    excluded = set(storage["excluded_files"])
    files = {p.relative_to(payload).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
             for p in sorted(payload.rglob("*")) if p.is_file() and p.name != "phase_manifest.json"
             and ".cache" not in p.parts and not p.name.endswith(".tmp") and "test_cache" not in p.parts
             and p.relative_to(payload).as_posix() not in excluded}
    manifest = {"format": 2, "storage_policy": storage, "phase": phase, "status": status, "config_sha256": digest(config),
                "prepared_manifest_digest": prepared_manifest_digest, "split_fingerprint": split_fingerprint,
                "source_hash": source_hash, "experiments": experiments, "files": files}
    write_json(payload / "phase_manifest.json", manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(temporary, "w", allowZip64=True, compresslevel=1) as archive:
        for name in [*files, "phase_manifest.json"]:
            compression = zipfile.ZIP_STORED if Path(name).suffix == ".zip" else zipfile.ZIP_DEFLATED
            archive.write(payload / name, f"Q2_{phase}/" + name, compress_type=compression)
    receipt = inspect_artifact(temporary, expected_phase=phase, expected_config=config, expected_source=source_hash, allow_incomplete=status == "FAILED")
    temporary.replace(output)
    write_json(output.with_suffix(".inspection.json"), receipt)
    return receipt
