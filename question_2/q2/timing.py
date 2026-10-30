"""Completion estimates from accumulated measurements, separated by model type."""
from datetime import datetime, timedelta, timezone
from .common import write_json


def estimate_remaining(config, records, overhead_minutes=15, target_world_sizes=None):
    measured = {}
    for name, record in records.items():
        model = name.split("_seed")[0]
        if target_world_sizes and model in target_world_sizes and record.get('world_size', 1) != target_world_sizes[model]:
            continue  # Old single-GPU epochs must not be averaged into DDP timing.
        values = [float(v) for v in record["epoch_seconds"] if v > 0]
        if values:
            measured.setdefault(model, []).extend(values)
    known_low = known_high = 0.0
    unknown, assumptions, remaining = [], [], []
    for model in ("baseline", "roberta_mean", "roberta_first"):
        limit = config["baseline" if model == "baseline" else "roberta"]["epochs"]
        source = model
        if model not in measured and model == "roberta_first" and "roberta_mean" in measured:
            source = "roberta_mean"
            assumptions.append("First-token runtime uses mean-pooling encoder timings as a proxy, with a widened range")
        values = measured.get(source)
        for seed in config["seeds"]:
            name = f"{model}_seed{seed}"
            if records.get(name, {}).get("complete"):
                continue
            already = len(records.get(name, {}).get("epoch_seconds", []))
            epochs = max(0, limit - already)
            remaining.append({"experiment": name, "maximum_remaining_epochs": epochs, "timing_source": source if values else None})
            if not values:
                unknown.append(name)
                continue
            low_factor, high_factor = (.65, 1.5) if source != model else (.8, 1.3)
            average = sum(values) / len(values)
            known_low += epochs * min(values) * low_factor
            known_high += epochs * max(max(values), average) * high_factor
    extra = overhead_minutes * 60 if remaining else 0
    low, high = known_low + extra, known_high + extra
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    result = {"timezone": "Asia/Calcutta (UTC+05:30)", "as_of": now.isoformat(), "remaining_runs": remaining,
        "unknown_model_timings": unknown, "overhead_minutes_assumed": overhead_minutes if remaining else 0,
        "known_work_range_hours_at_epoch_caps": [low / 3600, high / 3600],
        "whole_pipeline_estimate_available": not unknown,
        "assumptions": ["Uses observed epoch durations separately for each model, not a single epoch for all experiments",
            "Assumes remaining runs reach their epoch caps; early stopping may finish below this range",
            "Allows variability; download/export/metadata/evaluation overhead is a stated assumption, not a measurement",
            "Hardware changes can invalidate estimates"] + assumptions}
    if target_world_sizes:
        result['target_world_sizes'] = target_world_sizes
        result['assumptions'].append('Uses only measurements matching the target world size; historical unannotated runs used one training device')
    if not unknown:
        result["approximate_finish_range_ist_at_caps"] = [(now + timedelta(seconds=low)).isoformat(), (now + timedelta(seconds=high)).isoformat()]
    return result
