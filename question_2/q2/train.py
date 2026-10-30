"""Single-model, single-seed training. Never loads or scores the test split."""
import argparse
from contextlib import nullcontext
import gc
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from .common import (atomic_torch_save, capture_dependencies, digest, environment, file_hash,
                     load_config, logger, read_json, restore_rng, rng_state, seed_everything,
                     source_hash, write_json)
from .data import load_training_data, verify_prepared
from .metrics import metrics_from_confusion
from .models import make_model
from .validation import verify_completed_run


class EpochBatchSampler(Sampler):
    """Order determined by seed/epoch; resume skips batches without replaying them."""
    def __init__(self, size, batch_size, seed, epoch, start_batch=0):
        self.size, self.batch_size = size, batch_size
        self.seed, self.epoch, self.start_batch = seed, epoch, start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 1009 * self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for begin in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield order[begin:begin + self.batch_size]

    def __len__(self):
        return math.ceil(self.size / self.batch_size) - self.start_batch


def precision_settings(requested, device):
    if device.type != "cuda" or requested == "fp32":
        return None
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use auto or fp16")
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[requested]


def autocast_context(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def update_confusion(cm, labels, predictions):
    values = (labels.detach().cpu() * 5 + predictions.detach().cpu()).numpy()
    cm += np.bincount(values, minlength=25).reshape(5, 5)


def collator_for(model_name, prepared):
    if model_name == "baseline":
        return None
    from transformers import DataCollatorWithPadding, RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained(prepared / "tokenizer", local_files_only=True)
    return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")


@torch.inference_mode()
def evaluate_loader(model, loader, device, dtype, description="Validation", return_predictions=False):
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    loss_sum, count = 0.0, 0
    all_predictions, all_probabilities = [], []
    for batch in tqdm(loader, desc=description):
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        with autocast_context(device, dtype):
            logits = model(**batch)
            loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {description}")
        prediction = logits.argmax(-1)
        update_confusion(cm, labels, prediction)
        loss_sum += loss.item()
        count += len(labels)
        if return_predictions:
            all_predictions.extend(prediction.cpu().tolist())
            all_probabilities.extend(logits.float().softmax(-1).cpu().tolist())
    metrics = metrics_from_confusion(cm)
    metrics["loss"] = loss_sum / count
    if return_predictions:
        return metrics, np.array(all_predictions), np.array(all_probabilities)
    return metrics


def run_training(config, prepared, output, model_name, seed, allow_cpu=False, resume=True):
    prepared, output = Path(prepared), Path(output)
    if seed not in config["seeds"]:
        raise ValueError("Seed must be one of the planned seeds in config.json")
    output.mkdir(parents=True, exist_ok=True)
    log = logger(output / "training.log")
    manifest = verify_prepared(prepared, config)
    signature = digest({"config": config, "prepared_manifest": digest(manifest), "model": model_name,
                        "seed": seed, "source": source_hash()})
    run_file = output / "run.json"
    if run_file.exists() and read_json(run_file)["signature"] != signature:
        raise ValueError("Output directory belongs to different code/config/data; use a new directory")
    if (output / "complete.json").exists():
        completed = read_json(output / "complete.json")
        verify_completed_run(output, expected_signature=signature)
        log.info("Run already completed and verified; no retraining")
        return completed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("No CUDA GPU detected. Enable Kaggle GPU; full training on CPU is disabled.")
    seed_everything(seed)
    c = config["baseline"] if model_name == "baseline" else config["roberta"]
    t = config["training"]
    dtype = precision_settings(c["precision"], device)
    log.info("START model=%s seed=%d device=%s precision=%s full training data", model_name, seed, device, dtype)
    if torch.cuda.is_available():
        log.info("GPU=%s; visible GPUs=%d; this process uses one GPU", torch.cuda.get_device_name(0), torch.cuda.device_count())
        torch.cuda.reset_peak_memory_stats()
    train_data, val_data = load_training_data(prepared, model_name)
    expected = config["dataset"]["expected_sizes"]
    if len(train_data) != expected["train"] or len(val_data) != expected["validation"]:
        raise ValueError("Unexpected train/validation sizes; refusing accidental subset training")
    log.info("[progress] samples train=%d validation=%d | max epochs=%d | batch=%d accumulation=%d",
             len(train_data), len(val_data), c["epochs"], c["batch_size"], c["accumulation_steps"])
    collator = collator_for(model_name, prepared)
    model = make_model(model_name, config, prepared=prepared).to(device)
    metadata = {"vocab_size": model.embedding.num_embeddings} if model_name == "baseline" else {
        "encoder_config": model.encoder.config.to_dict()}
    parameters = sum(p.numel() for p in model.parameters())
    batches_per_epoch = math.ceil(len(train_data) / c["batch_size"])
    updates_per_epoch = math.ceil(batches_per_epoch / c["accumulation_steps"])
    if model_name == "baseline":
        # Include embeddings now, even while frozen; unfreezing does not reset Adam.
        optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
        scheduler = None
    else:
        from transformers import get_linear_schedule_with_warmup
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": c["weight_decay"]},
                                       {"params": no_decay, "weight_decay": 0.0}], lr=c["learning_rate"])
        total_updates = updates_per_epoch * c["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_updates * c["warmup_ratio"]), total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    run = {"signature": signature, "config": config, "seed": seed, "model": model_name,
           "metadata": metadata, "parameters": parameters, "environment": environment(),
           "prepared_manifest": digest(manifest), "precision": str(dtype or torch.float32),
           "planned_optimizer_updates": updates_per_epoch * c["epochs"], "source_hash": source_hash()}
    prior_run = read_json(run_file) if run_file.exists() else None
    run["sessions"] = (prior_run.get("sessions", [prior_run["environment"]]) if prior_run else []) + [run["environment"]]
    write_json(run_file, run)
    capture_dependencies(output)
    state = {"epoch": 0, "next_batch": 0, "optimizer_steps": 0, "best_f1": -1.0,
             "best_epoch": None, "bad_epochs": 0, "history": [], "elapsed_seconds": 0.0,
             "partial": {"loss_sum": 0.0, "count": 0, "confusion": np.zeros((5, 5), dtype=np.int64)}}
    last = output / "last.pt"
    if last.exists():
        if not resume:
            raise ValueError("Recovery checkpoint exists. Resume or choose a fresh output directory.")
        # Only load checkpoints produced by this project/from your own attached output.
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise ValueError("Recovery checkpoint signature mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        restore_rng(checkpoint["rng"])
        del checkpoint
        log.info("RESUME epoch=%d next batch=%d optimizer steps=%d best F1=%.4f", state["epoch"] + 1,
                 state["next_batch"], state["optimizer_steps"], state["best_f1"])
    session_start = time.monotonic()
    elapsed_before_session = state["elapsed_seconds"]
    stop_requested = [False]
    def request_stop(signum, frame):
        stop_requested[0] = True
        log.warning("Termination requested; saving at next optimizer boundary")
    previous_handler = signal.signal(signal.SIGTERM, request_stop)

    def save_recovery():
        state["elapsed_seconds"] = elapsed_before_session + time.monotonic() - session_start
        atomic_torch_save({"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict(),
                           "state": state, "rng": rng_state()}, last)
        write_json(output / "status.json", {"status": "training", "epoch": state["epoch"] + 1,
                   "next_batch": state["next_batch"], "optimizer_steps": state["optimizer_steps"],
                   "best_validation_macro_f1": state["best_f1"], "elapsed_seconds": state["elapsed_seconds"]})
        log.info("Recovery saved: epoch=%d next batch=%d", state["epoch"] + 1, state["next_batch"])

    try:
        while state["epoch"] < c["epochs"] and state["bad_epochs"] < c["patience"]:
            epoch = state["epoch"]
            epoch_start = time.monotonic()
            epoch_elapsed_before = state["partial"].get("elapsed_seconds", 0.0)
            if model_name == "baseline":
                model.embedding.weight.requires_grad_(epoch >= 1)
                log.info("Embedding layer %s", "FROZEN (first epoch)" if epoch == 0 else "UNFROZEN")
            sampler = EpochBatchSampler(len(train_data), c["batch_size"], seed, epoch, state["next_batch"])
            loader_generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collator,
                                num_workers=t["num_workers"], pin_memory=device.type == "cuda", generator=loader_generator)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            partial = state["partial"]
            bar = tqdm(loader, total=len(loader), desc=f"{model_name} seed={seed} epoch={epoch + 1}/{c['epochs']}")
            for index, batch in enumerate(bar, start=state["next_batch"]):
                batch = move_batch(batch, device)
                labels = batch.pop("labels")
                # Correct sample weighting even for the last incomplete accumulation group.
                group_begin = (index // c["accumulation_steps"]) * c["accumulation_steps"] * c["batch_size"]
                group_samples = min(c["accumulation_steps"] * c["batch_size"], len(train_data) - group_begin)
                with autocast_context(device, dtype):
                    logits = model(**batch)
                    batch_loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
                    loss = batch_loss / group_samples
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; inspect log and recover last saved checkpoint")
                scaler.scale(loss).backward()
                partial["loss_sum"] += batch_loss.item()
                partial["count"] += len(labels)
                update_confusion(partial["confusion"], labels, logits.argmax(-1))
                boundary = (index + 1) % c["accumulation_steps"] == 0 or index + 1 == batches_per_epoch
                if boundary:
                    scaler.unscale_(optimizer)
                    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
                    if not torch.isfinite(gradient_norm) and not scaler.is_enabled():
                        raise FloatingPointError("Nonfinite gradients")
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    update_applied = scaler.get_scale() >= scale_before
                    if scheduler and update_applied:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    state["optimizer_steps"] += int(update_applied)
                    state["next_batch"] = index + 1
                    if stop_requested[0] or (update_applied and state["optimizer_steps"] % t["checkpoint_every_optimizer_steps"] == 0):
                        partial["elapsed_seconds"] = epoch_elapsed_before + time.monotonic() - epoch_start
                        save_recovery()
                    if stop_requested[0]:
                        write_json(output / "status.json", {"status": "interrupted", "next_batch": index + 1})
                        return {"status": "interrupted"}
                if (index + 1) % t["log_every_batches"] == 0 or index + 1 == batches_per_epoch:
                    running = metrics_from_confusion(partial["confusion"])
                    elapsed = time.monotonic() - session_start
                    memory = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                    bar.set_postfix(loss=f"{partial['loss_sum']/partial['count']:.4f}", f1=f"{running['macro_f1']:.4f}")
                    log.info("epoch=%d batch=%d/%d loss=%.4f train F1=%.4f lr=%.3g elapsed=%.1f min peak GPU=%.2f GiB",
                             epoch + 1, index + 1, batches_per_epoch, partial["loss_sum"] / partial["count"],
                             running["macro_f1"], optimizer.param_groups[0]["lr"], elapsed / 60, memory)
            if partial["count"] != len(train_data):
                raise ValueError("Training epoch did not cover every training record")
            val_loader = DataLoader(val_data, batch_size=c["eval_batch_size"], shuffle=False, collate_fn=collator,
                                    num_workers=t["num_workers"], pin_memory=device.type == "cuda",
                                    generator=torch.Generator().manual_seed(seed))
            validation = evaluate_loader(model, val_loader, device, dtype)
            training = metrics_from_confusion(partial["confusion"])
            training["loss"] = partial["loss_sum"] / partial["count"]
            elapsed_total = elapsed_before_session + time.monotonic() - session_start
            epoch_seconds = epoch_elapsed_before + time.monotonic() - epoch_start
            remaining_seconds = epoch_seconds * (c["epochs"] - epoch - 1)
            log.info("[progress] epoch_seconds=%.1f approximate remaining at epoch cap=%.1f min; early stopping may shorten",
                     epoch_seconds, remaining_seconds / 60)
            row = {"epoch": epoch + 1, "train": training, "validation": validation, "epoch_seconds": epoch_seconds,
                   "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed_total}
            state["history"].append(row)
            improved = validation["macro_f1"] > state["best_f1"] + t["min_delta"]
            if improved:
                state["best_f1"], state["best_epoch"], state["bad_epochs"] = validation["macro_f1"], epoch + 1, 0
                atomic_torch_save({"signature": signature, "model": model.state_dict(), "run": run,
                                   "best_epoch": epoch + 1, "validation": validation}, output / "best.pt")
            else:
                state["bad_epochs"] += 1
            write_json(output / "history.json", state["history"])
            log.info("EPOCH %d COMPLETE | train loss=%.4f F1=%.4f | val loss=%.4f F1=%.4f | best epoch=%d | patience=%d/%d",
                     epoch + 1, training["loss"], training["macro_f1"], validation["loss"], validation["macro_f1"],
                     state["best_epoch"], state["bad_epochs"], c["patience"])
            state["epoch"], state["next_batch"] = epoch + 1, 0
            state["partial"] = {"loss_sum": 0.0, "count": 0, "confusion": np.zeros((5, 5), dtype=np.int64)}
            save_recovery()
        result = {"status": "complete", "signature": signature, "model": model_name, "seed": seed,
                  "best_epoch": state["best_epoch"], "best_validation_macro_f1": state["best_f1"],
                  "epochs_completed": state["epoch"], "elapsed_seconds": state["elapsed_seconds"],
                  "best_sha256": file_hash(output / "best.pt"), "parameters": parameters,
                  "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0}
        result["evidence_sha256"] = {name: file_hash(output / name) for name in
                                     ("run.json", "history.json", "best.pt", "environment-freeze.txt")}
        write_json(output / "complete.json", result)
        write_json(output / "status.json", result)
        log.info("TRAINING COMPLETE | %s seed %d | %.2f hours | best val F1=%.4f | test untouched",
                 model_name, seed, result["elapsed_seconds"] / 3600, state["best_f1"])
        return result
    except Exception:
        # last.pt is always an optimizer boundary; do not overwrite with partial gradients.
        log.exception("RUN FAILED. Keep last.pt AND best.pt for recovery; see README.")
        write_json(output / "failure.json", {"environment": environment(), "recovery_available": last.exists()})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["baseline", "roberta_mean", "roberta_first"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-resume", action="store_true", help="Reject an existing recovery checkpoint")
    parser.add_argument("--allow-cpu", action="store_true", help="For developer correctness tests, not full Kaggle training")
    args = parser.parse_args()
    result = run_training(load_config(args.config), args.prepared, args.output, args.model, args.seed, args.allow_cpu, resume=not args.no_resume)
    if result["status"] != "complete":
        raise SystemExit("Run interrupted and recovery saved; this is not a completed experiment.")


if __name__ == "__main__":
    main()
