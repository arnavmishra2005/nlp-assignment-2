"""Few-batch correctness exercise on training/validation only; never full results."""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from .common import atomic_torch_save, digest, load_config, logger, read_json, rng_state, seed_everything, write_json
from .data import load_training_data, verify_prepared
from .models import make_model
from .train import autocast_context, collator_for, evaluate_loader, move_batch, precision_settings


def run_smoke(config, prepared, output, allow_cpu=False, batches=2):
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = logger(output / "smoke.log")
    verify_prepared(prepared, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("The Kaggle smoke phase requires a GPU")
    seed = config["seeds"][0]
    # Two examples per class, chosen without consulting any test records.
    frames = {split: pd.read_parquet(prepared / f"{split}.parquet") for split in ("train", "validation")}
    indices = {split: [int(i) for label in range(5) for i in frame.index[frame.label == label][:2]]
               for split, frame in frames.items()}
    summaries = []
    for name in ("baseline", "roberta_mean"):
        seed_everything(seed)
        folder = output / name
        folder.mkdir(exist_ok=True)
        datasets = load_training_data(prepared, name)
        collate = collator_for(name, prepared)
        loaders = [DataLoader(Subset(data, indices[split]), batch_size=2, shuffle=False, collate_fn=collate)
                   for data, split in zip(datasets, ("train", "validation"))]
        model = make_model(name, config, prepared=prepared).to(device)
        c = config["baseline"] if name == "baseline" else config["roberta"]
        dtype = precision_settings(c["precision"], device)
        optimizer = (torch.optim.Adam if name == "baseline" else torch.optim.AdamW)(model.parameters(), lr=c["learning_rate"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / 2))
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=1024)
        model.train()
        losses = []
        for index, batch in enumerate(loaders[0]):
            if index >= batches:
                break
            if name == "baseline":
                model.embedding.weight.requires_grad_(index > 0)
            batch = move_batch(batch, device)
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                logits = model(**batch)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite smoke loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite smoke gradients")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.item()))
            log.info("SMOKE ONLY | %s | batch=%d/%d loss=%.4f", name, index + 1, batches, losses[-1])
        metrics, predictions, probabilities = evaluate_loader(model, loaders[1], device, dtype,
                description="SMOKE validation only", return_predictions=True)
        metadata = {"vocab_size": model.embedding.num_embeddings} if name == "baseline" else {
            "encoder_config": model.encoder.config.to_dict()}
        checkpoint = folder / "smoke_checkpoint.pt"
        atomic_torch_save({"purpose": "SMOKE_ONLY", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
            "config": config, "metadata": metadata, "batches": len(losses)}, checkpoint)
        # Release the original model before constructing its independent reload.
        del model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restored = make_model(name, config, metadata=metadata).to(device)
        restored.load_state_dict(saved["model"])
        del saved
        reloaded_metrics, reloaded_predictions, reloaded_probabilities = evaluate_loader(restored, loaders[1], device, dtype,
                description="SMOKE checkpoint reload", return_predictions=True)
        np.testing.assert_array_equal(predictions, reloaded_predictions)
        np.testing.assert_allclose(probabilities, reloaded_probabilities, atol=1e-5, rtol=1e-4)
        selected = frames["validation"].iloc[indices["validation"]]
        rows = pd.DataFrame({"id": selected.id.to_numpy(), "true_stars": selected.label.to_numpy() + 1,
                             "predicted_stars": predictions + 1})
        for label in range(5):
            rows[f"probability_{label+1}"] = probabilities[:, label]
        rows.to_csv(folder / "predictions.csv", index=False)
        write_json(folder / "metrics.json", {"purpose": "SMOKE_ONLY", "metrics": metrics,
                   "train_losses": losses, "checkpoint_reload_predictions_match": True})
        summaries.append({"model": name, "batches": len(losses), "validation_samples": len(selected),
                          "checkpoint_reload_predictions_match": True})
        del restored
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {"status": "COMPLETE", "purpose": "SMOKE_ONLY_NOT_ASSIGNMENT_RESULTS", "models": summaries,
              "config_sha256": digest(config), "test_accessed": False}
    write_json(output / "SMOKE_COMPLETE.json", result)
    log.info("SMOKE_CHECK_PASSED. Correctness only; no performance target evaluated.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()
    if args.batches < 2:
        parser.error("Use at least two batches to exercise embedding unfreezing")
    run_smoke(load_config(args.config), args.prepared, args.output, args.allow_cpu, args.batches)
