"""Freeze experiments, then evaluate the main models once on the held-out test split."""
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import (digest, environment, file_hash, load_config, logger, read_json,
                     source_hash, write_json)
from .data import ClassicDataset, encode_classic, load_split, verify_prepared
from .models import make_model, load_trained
from .validation import verify_completed_run
from .execution import execution_details
from .train import collator_for, evaluate_loader, precision_settings

MODELS = ("baseline", "roberta_mean", "roberta_first")


def expected_runs(config, runs):
    return [(model, seed, Path(runs) / f"{model}_seed{seed}") for model in MODELS for seed in config["seeds"]]


def validate_runs(config, prepared, runs):
    manifest = verify_prepared(prepared, config)
    records = []
    for model, seed, directory in expected_runs(config, runs):
        verify_completed_run(directory)
        run = read_json(directory / "run.json")
        complete = read_json(directory / "complete.json")
        from .compatibility import compatible_prerequisite
        source_ok = run["source_hash"] == source_hash() or compatible_prerequisite(
            {"phase": f"{model}_seed{seed}", "source_hash": run["source_hash"]}, source_hash())
        if run["config"] != config or not source_ok:
            raise ValueError(f"Different config/source in {directory}; do not mix experiments")
        if run["prepared_manifest"] != digest(manifest) or run["model"] != model or run["seed"] != seed:
            raise ValueError(f"Data/model/seed mismatch in {directory}")
        if complete["status"] != "complete" or complete["signature"] != run["signature"]:
            raise ValueError(f"Incomplete run {directory}")
        checksum = file_hash(directory / "best.pt")
        if checksum != complete["best_sha256"]:
            raise ValueError(f"Corrupt or replaced checkpoint {directory}")
        records.append({"model": model, "seed": seed, "signature": run["signature"],
                        "checkpoint_sha256": checksum, 'training_execution': execution_details(run)})
    return records


def freeze_protocol(config, records, final):
    protocol = {"config": config, "runs": records, "qualitative_seed": config["qualitative_seed"],
                "test_models": ["baseline", "roberta_mean"],
                "ablation_selection": "validation only; first-token model not tested",
                "decision": "All settings fixed before test access; do not tune after seeing test results"}
    path = final / "FROZEN_PROTOCOL.json"
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError("Test protocol already frozen for different experiments. Do not retune on test results.")
    else:
        write_json(path, protocol)
    return protocol


def evaluate_final(config, prepared, runs, final, allow_cpu=False):
    from datasets import Dataset as HFDataset
    from transformers import RobertaTokenizerFast
    prepared, runs, final = Path(prepared), Path(runs), Path(final)
    final.mkdir(parents=True, exist_ok=True)
    log = logger(final / "evaluation.log")
    records = validate_runs(config, prepared, runs)
    protocol = freeze_protocol(config, records, final)
    protocol_hash = digest(protocol)
    main_records = [r for r in records if r["model"] != "roberta_first"]
    # Completed predictions are the single source for regenerated plots/reports.
    pending = []
    for record in main_records:
        name = f"{record['model']}_seed{record['seed']}"
        metric_file = final / "metrics" / f"{name}.json"
        prediction_file = final / "predictions" / f"{name}.csv"
        if metric_file.exists():
            metric = read_json(metric_file)
            if metric["protocol_hash"] != protocol_hash or file_hash(prediction_file) != metric["prediction_sha256"]:
                raise ValueError(f"Cached evaluation changed: {name}")
            log.info("Using recorded test predictions for %s; not scoring again", name)
        else:
            pending.append(record)
    if pending:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu" and not allow_cpu:
            raise RuntimeError("Enable GPU for final inference, or explicitly use --allow-cpu")
        cache = final / "test_cache"
        cache.mkdir(exist_ok=True)
        test_path = cache / "test.parquet"
        if not test_path.exists():
            # This is the only function in the training workflow that requests test.
            log.info("FINAL TEST ACCESS: all six training runs verified and frozen")
            test = load_split(config, "test", log)
            train_ids = set(pd.read_parquet(prepared / "train.parquet", columns=["id"]).id)
            validation_ids = set(pd.read_parquet(prepared / "validation.parquet", columns=["id"]).id)
            if set(test.id) & (train_ids | validation_ids):
                raise ValueError("Test review IDs overlap with training/validation")
            temporary_test = cache / "test.parquet.tmp"
            test.to_parquet(temporary_test, index=False)
            write_json(cache / "audit.json", {"protocol_hash": protocol_hash, "test_sha256": file_hash(temporary_test),
                       "environment": environment(), "rows": len(test), "test_ids_disjoint": True})
            os.replace(temporary_test, test_path)
        else:
            audit = read_json(cache / "audit.json")
            if audit["protocol_hash"] != protocol_hash or audit["test_sha256"] != file_hash(test_path):
                raise ValueError("Cached test data/protocol changed")
            test = pd.read_parquet(test_path)
        pd.DataFrame({"id": test.id, "true_stars": test.label + 1}).to_csv(final / "test_ids.csv", index=False)
        vocab = read_json(prepared / "vocab.json")
        ids, lengths, stats = encode_classic(test.text.tolist(), vocab, config["baseline"]["max_length"])
        baseline_data = ClassicDataset({"input_ids": ids, "lengths": lengths, "labels": test.label.to_numpy()})
        tokenizer = RobertaTokenizerFast.from_pretrained(prepared / "tokenizer", local_files_only=True)
        roberta_data = HFDataset.from_dict({"text": test.text.tolist(), "labels": test.label.tolist()})
        roberta_data = roberta_data.map(lambda batch: tokenizer(batch["text"], truncation=True,
                       max_length=config["roberta"]["max_length"], padding=False), batched=True, remove_columns=["text"])
        for record in pending:
            model_name, seed = record["model"], record["seed"]
            name = f"{model_name}_seed{seed}"
            model, run = load_trained(runs / name, device)
            c = config["baseline"] if model_name == "baseline" else config["roberta"]
            dataset = baseline_data if model_name == "baseline" else roberta_data
            loader = DataLoader(dataset, batch_size=c["eval_batch_size"], shuffle=False,
                                collate_fn=collator_for(model_name, prepared),
                                num_workers=config["training"]["num_workers"], pin_memory=device.type == "cuda")
            metrics, prediction, probabilities = evaluate_loader(model, loader, device,
                    precision_settings(c["precision"], device), description=f"FINAL TEST {name}", return_predictions=True)
            frame = pd.DataFrame({"id": test.id, "true_stars": test.label + 1, "predicted_stars": prediction + 1})
            for label in range(5):
                frame[f"probability_{label + 1}"] = probabilities[:, label]
            pred_path = final / "predictions" / f"{name}.csv"
            pred_path.parent.mkdir(exist_ok=True)
            temporary = pred_path.with_suffix(".tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, pred_path)
            write_json(final / "metrics" / f"{name}.json", {"model": model_name, "seed": seed,
                       "protocol_hash": protocol_hash, "prediction_sha256": file_hash(pred_path),
                       "metrics": metrics, "baseline_test_preprocessing": stats if model_name == "baseline" else None})
            log.info("%s | accuracy=%.4f macro F1=%.4f MAE=%.4f", name,
                     metrics["accuracy"], metrics["macro_f1"], metrics["mae"])
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log.info("All final predictions saved; future analysis uses these cached predictions")
    from .analysis import create_analysis
    create_analysis(config, prepared, runs, final, log)
    write_json(final / "EVALUATION_COMPLETE.json", {"protocol_hash": protocol_hash, "environment": environment(),
               "note": "Review qualitative annotations and complete the report; do not tune using test results"})
    log.info("FINAL EVALUATION AND ANALYSIS COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    evaluate_final(load_config(args.config), args.prepared, args.runs, args.output, args.allow_cpu)


if __name__ == "__main__":
    main()
