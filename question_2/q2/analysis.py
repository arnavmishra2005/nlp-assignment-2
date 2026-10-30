"""Plots, seed statistics, qualitative examples and gradient-times-input importance."""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import file_hash, read_json, write_json
from .metrics import classification_metrics
from .execution import execution_details, execution_note

SCALARS = ["accuracy", "macro_f1", "mae", "adjacent_error_rate", "distant_error_rate",
           "adjacent_share_of_errors", "distant_share_of_errors", "extreme_1_vs_5_error_rate"]


def save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def confusion_plot(matrix, title, output):
    cm = np.asarray(matrix, dtype=float)
    support = cm.sum(1, keepdims=True)
    normalized = np.divide(cm, support, out=np.zeros_like(cm), where=support > 0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{normalized[i,j]:.2f}", ha="center", va="center",
                    color="white" if normalized[i, j] > .5 else "black")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=range(1, 6), yticklabels=range(1, 6),
           xlabel="Predicted stars", ylabel="True stars", title=title)
    fig.colorbar(im, ax=ax, label="Proportion within true rating")
    save_figure(fig, output)


def training_plots(config, runs, figures):
    for model in ("baseline", "roberta_mean", "roberta_first"):
        for seed in config["seeds"]:
            name = f"{model}_seed{seed}"
            history = read_json(runs / name / "history.json")
            epochs = [r["epoch"] for r in history]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            for axis, metric, title in zip(axes, ("loss", "macro_f1"), ("Cross-entropy loss", "Macro F1")):
                for split in ("train", "validation"):
                    axis.plot(epochs, [r[split][metric] for r in history], marker="o", label=split)
                axis.set(xlabel="Epoch", ylabel=title, xticks=epochs)
                axis.legend()
                axis.grid(alpha=.2)
            fig.suptitle(name + " (training metrics accumulated with dropout active)")
            save_figure(fig, figures / f"{name}_curves.png")


def aggregate(rows, metrics):
    table = pd.DataFrame(rows)
    result = []
    for model, group in table.groupby("model", sort=False):
        row = {"model": model, "seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        result.append(row)
    return pd.DataFrame(result)


def markdown_table(frame, columns=None):
    if columns:
        frame = frame[columns]
    def cell(value):
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def qualitative_examples(config, final, log):
    seed = config["qualitative_seed"]
    baseline = pd.read_csv(final / "predictions" / f"baseline_seed{seed}.csv")
    transformer = pd.read_csv(final / "predictions" / f"roberta_mean_seed{seed}.csv")
    if not baseline.id.equals(transformer.id) or not baseline.true_stars.equals(transformer.true_stars):
        raise ValueError("Predictions are not aligned across models")
    test = pd.read_parquet(final / "test_cache" / "test.parquet")
    if test.id.tolist() != baseline.id.tolist():
        raise ValueError("Prediction/test order mismatch")
    frame = test[["id", "title", "body", "text"]].copy()
    frame["true_stars"] = baseline.true_stars
    frame["baseline_stars"] = baseline.predicted_stars
    frame["roberta_stars"] = transformer.predicted_stars
    baseline_ok = frame.true_stars == frame.baseline_stars
    transformer_ok = frame.true_stars == frame.roberta_stars
    groups = [("baseline_wrong_roberta_right", ~baseline_ok & transformer_ok, 5),
              ("baseline_right_roberta_wrong", baseline_ok & ~transformer_ok, 2)]
    examples, counts = [], {}
    for name, mask, requested in groups:
        candidates = frame[mask].sort_values("id")
        counts[name] = {"available": len(candidates), "requested": requested, "selected": min(requested, len(candidates))}
        selected = candidates.head(requested).copy()
        selected["comparison"] = name
        selected["human_error_category"] = ""
        selected["human_analysis"] = ""
        examples.append(selected)
    selected = pd.concat(examples, ignore_index=True)
    selected.to_csv(final / "qualitative_examples.csv", index=False)
    write_json(final / "qualitative_selection.json", {"seed": seed, "selection": "ascending review ID, first requested count",
               "counts": counts, "note": "Fill human_error_category and human_analysis in a separate annotated copy"})
    lines = ["# Qualitative examples", "", f"Fixed seed {seed}; examples selected by ascending review ID, not by manual preference.",
             "", "Analyze negation, sarcasm, mixed sentiment, long reviews, misspellings and rating ambiguity where supported by text.",
             "These are hypotheses: a prediction difference alone does not establish the model's reasoning.", ""]
    for _, row in selected.iterrows():
        lines += [f"## {row.id} - {row.comparison}", "", f"True: {row.true_stars}; BiGRU: {row.baseline_stars}; RoBERTa: {row.roberta_stars}",
                  "", "> " + row.text.replace("\n", "\n> "), "", "**Human analysis:** TODO after reading this example.", ""]
    (final / "qualitative_examples.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Qualitative selection: %s", counts)
    preferred = selected[selected.comparison == "baseline_wrong_roberta_right"]
    chosen = preferred.iloc[0] if len(preferred) else frame.sort_values("id").iloc[0]
    return chosen


def token_importance(config, prepared, runs, final, example, log):
    from transformers import RobertaTokenizerFast
    from .evaluate import load_trained
    signature = {"review_id": example["id"], "seed": config["qualitative_seed"],
                 "checkpoint": file_hash(runs / f"roberta_mean_seed{config['qualitative_seed']}" / "best.pt"),
                 "method": "abs(sum_embedding_dimensions(embedding * gradient_of_predicted_class_logit))"}
    cache = final / "token_importance.json"
    if cache.exists() and read_json(cache)["signature"] == signature:
        log.info("Token importance already recorded; reusing")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_trained(runs / f"roberta_mean_seed{config['qualitative_seed']}", device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = RobertaTokenizerFast.from_pretrained(prepared / "tokenizer", local_files_only=True)
    encoded = tokenizer(example["text"], truncation=True, max_length=config["roberta"]["max_length"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in encoded.items()}
    embedded = model.encoder.get_input_embeddings()(inputs["input_ids"]).detach().requires_grad_(True)
    states = model.encoder(inputs_embeds=embedded, attention_mask=inputs["attention_mask"], return_dict=True).last_hidden_state
    logits = model.pool_and_classify(states, inputs["attention_mask"])
    predicted = int(logits.argmax(-1).item())
    gradient = torch.autograd.grad(logits[0, predicted], embedded)[0]
    importance = (gradient * embedded).sum(-1).abs()[0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    importance /= max(float(importance.sum()), 1e-12)
    table = pd.DataFrame({"position": range(len(tokens)), "token": tokens, "importance": importance})
    table.to_csv(final / "token_importance.csv", index=False)
    fig, axes = plt.subplots(math.ceil(len(tokens) / 32), 1, figsize=(13, 1.8 * math.ceil(len(tokens) / 32)), squeeze=False)
    for row, axis in enumerate(axes[:, 0]):
        begin, end = row * 32, min((row + 1) * 32, len(tokens))
        axis.imshow(importance[None, begin:end], cmap="Oranges", aspect="auto", vmin=0, vmax=max(importance.max(), 1e-12))
        labels = [t.replace("\u0120", "_").replace("\u010a", "[newline]") for t in tokens[begin:end]]
        axis.set(xticks=range(end - begin), xticklabels=labels, yticks=[], ylabel=f"{begin}:{end}")
        axis.tick_params(axis="x", labelrotation=60, labelsize=7)
    fig.suptitle(f"Gradient x input importance | {example['id']} | predicted {predicted + 1} stars\n"
                 "Absolute normalized scores; includes special tokens; not a causal explanation")
    save_figure(fig, final / "figures" / "roberta_token_importance.png")
    write_json(cache, {"signature": signature, "true_stars": int(example["true_stars"]),
               "predicted_stars": predicted + 1, "tokens": tokens, "importance": importance.tolist(),
               "notes": f"Evaluation mode, float32, fixed predicted-class logit; tokenization truncated at {config['roberta']['max_length']}; not causal evidence"})
    log.info("Token importance saved for %s", example["id"])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def report_draft(config, prepared, runs, final, test_summary, val_summary):
    stats = read_json(prepared / "data_statistics.json")
    selection = read_json(final / "qualitative_selection.json")
    table = test_summary.copy()
    display_rows = []
    for _, row in table.iterrows():
        display_rows.append({"Model": row.model,
            **{metric: f"{row[metric + '_mean']:.4f} +/- {row[metric + '_std']:.4f}"
               for metric in ("accuracy", "macro_f1", "mae")}})
    main = markdown_table(pd.DataFrame(display_rows))
    ablation = markdown_table(val_summary, ["model", "macro_f1_mean", "macro_f1_std", "mae_mean", "mae_std"])
    baseline = test_summary[test_summary.model == "baseline"].iloc[0]
    roberta = test_summary[test_summary.model == "roberta_mean"].iloc[0]
    details = pd.read_csv(final / 'run_details.csv')
    topology_table = markdown_table(details, ['model', 'seed', 'strategy', 'world_size', 'effective_batch_size'])
    topology_note = execution_note(details.to_dict('records'))
    text = f"""# Q2 report draft - fill analysis and format to at most six pages

**Status:** measured tables generated from completed runs; human interpretation and final formatting pending.

## 1. Data and reproducibility

Use the specified SetFit Amazon English review corpus: 200,000 training, 5,000 validation, 5,000 test reviews, five balanced ratings. SetFit exposes review bodies only. Recover titles from `buruzaemon/amazon_reviews_multi` using exact matching review IDs, body text, language and stars. Concatenate title, newline, body for both models. Original null titles are treated as empty and counted: train {stats.get('train', {}).get('missing_titles', 'not recorded')}, validation {stats.get('validation', {}).get('missing_titles', 'not recorded')}; no content is invented. Dataset/model revisions are pinned in `config.json`; file checksums and environment are recorded. The official splits are retained. Exact train/validation text overlap: {stats['exact_train_validation_text_overlap']}.

Classic path: HTML unescape, lowercase, remove HTML/URLs, regex word/contraction/punctuation tokenization; vocabulary fitted on training only, minimum frequency {config['baseline']['min_frequency']}, maximum {config['baseline']['max_vocab']}; PAD=0, UNK=1, {config['baseline']['max_length']}-token limit. Actual vocabulary: {stats['vocabulary_size']}; GloVe vocabulary coverage: {stats['embeddings']['vocabulary_coverage']:.2%}. OOV vectors use fixed seed 2026. RoBERTa receives original concatenated text with byte-level BPE, special tokens, {config['roberta']['max_length']}-token truncation and dynamic batch padding to multiples of eight.

## 2. Architectures and optimization

BiGRU: GloVe 6B {config['baseline']['embedding_dim']}d, two bidirectional layers, {config['baseline']['hidden_dim']} hidden units per direction, {config['baseline']['attention_dim']}d additive attention, masked padding, dropout {config['baseline']['dropout']}, five logits. Packed sequences prevent padded tokens affecting the backward GRU. Freeze embeddings in epoch one, unfreeze afterwards. Adam lr={config['baseline']['learning_rate']}, batch={config['baseline']['batch_size']}, maximum {config['baseline']['epochs']} epochs, patience {config['baseline']['patience']}.

RoBERTa-base: fine-tune the entire encoder, masked mean of the last hidden states over non-padding tokens (including special tokens), dropout {config['roberta']['dropout']}, five logits. AdamW lr={config['roberta']['learning_rate']}, weight decay {config['roberta']['weight_decay']} except biases/LayerNorm, {config['roberta']['warmup_ratio']:.0%} linear warmup then decay, batch={config['roberta']['batch_size']} with accumulation={config['roberta']['accumulation_steps']}, maximum {config['roberta']['epochs']} epochs, patience {config['roberta']['patience']}. Precision is bf16 when supported, otherwise fp16 on CUDA. Both use cross-entropy, gradient clipping at {config['training']['gradient_clip']}, validation Macro F1 checkpoint selection, seeds {config['seeds']}, and min improvement {config['training']['min_delta']}. Actual settings/hardware/timing/best epochs are in `run_details.csv` and run manifests. Early stopping can produce different actual epoch counts.

{topology_note}

{topology_table}

## 3. Quantitative results

Final test results below are **mean +/- sample standard deviation across seeds**, not confidence intervals. Predictions use argmax; map labels 0-4 to stars 1-5. MAE respects ordinal distance. The test protocol was frozen before test access; cached predictions generate all analysis.

{main}

Observed mean RoBERTa minus BiGRU Macro F1: {roberta.macro_f1_mean - baseline.macro_f1_mean:+.4f}; MAE: {roberta.mae_mean - baseline.mae_mean:+.4f}. These descriptive differences are not significance tests.

Include per-class F1 (`per_class_summary.csv`), normalized confusion matrices and rating-distance rates (`test_summary.csv`). Adjacent means |predicted-true|=1; distant means >=2. Error rates divide by all reviews; shares divide by misclassified reviews. Discuss whether improvements differ across ratings and whether errors remain mostly adjacent. **TODO: interpret actual class errors and ordinal behavior.**

## 4. Ablation

Mean versus first-token pooling; model/data/optimizer settings and paired seeds held fixed, with execution topology documented above. Main mean-pooling runs are reused. The ablation is evaluated on validation only; first-token models are not scored on test.

{ablation}

Include `ablation_paired.csv`; compare per-seed validation differences and variability. **TODO: explain the result without treating two seeds as strong statistical evidence.**

## 5. Qualitative comparison and visualization

Use fixed seed {config['qualitative_seed']}. Select examples by ascending review ID. Availability/selection: {selection['counts']}. Include five baseline-fails/RoBERTa-succeeds examples and two reverse examples when available. If fewer exist, report the actual count. Full texts and predictions are in `qualitative_examples.md`; annotate a copy of the CSV with error categories and evidence-based hypotheses (negation, sarcasm, mixed sentiment, long reviews, misspellings, or ambiguity).

`figures/roberta_token_importance.png` visualizes absolute gradient x input for the fixed model's predicted-class logit. Scores sum to one across retained tokens, including special tokens. This is local sensitivity, not causal proof or an attention alignment. **TODO: discuss the displayed tokens, truncation and the selected example; do not assert sarcasm or reasoning without textual evidence.**

## 6. Limitations and conclusion

**TODO after results:** discuss rating ambiguity, truncation, OOV words, domain variation, compute budget, early stopping and only two seeds. RoBERTa benefits from extensive external pretraining; architecture alone is not isolated by this comparison. State which model performed better using the measured metrics, without claiming improvements in advance. Include actual contribution details in the combined report.

## References

- Assignment PDF, pp. 5-7.
- https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en
- https://huggingface.co/datasets/buruzaemon/amazon_reviews_multi
- https://arxiv.org/abs/2010.02573 (Multilingual Amazon Reviews Corpus)
- https://nlp.stanford.edu/projects/glove/
- https://huggingface.co/FacebookAI/roberta-base
- https://arxiv.org/abs/1907.11692 (RoBERTa)
"""
    (final / "REPORT_DRAFT.md").write_text(text, encoding="utf-8")


def create_analysis(config, prepared, runs, final, log):
    figures = final / "figures"
    figures.mkdir(exist_ok=True)
    training_plots(config, runs, figures)
    validation_rows, run_details = [], []
    for model in ("baseline", "roberta_mean", "roberta_first"):
        for seed in config["seeds"]:
            path = runs / f"{model}_seed{seed}"
            completion, run = read_json(path / "complete.json"), read_json(path / "run.json")
            history = read_json(path / "history.json")
            best = next(r for r in history if r["epoch"] == completion["best_epoch"])
            validation_rows.append({"model": model, "seed": seed,
                                    **{key: best["validation"][key] for key in SCALARS}})
            run_details.append({"model": model, "seed": seed, "parameters": run["parameters"],
                "gpu": run["environment"]["gpu"], "precision": run["precision"],
                "best_epoch": completion["best_epoch"], "epochs_completed": completion["epochs_completed"],
                "training_hours": completion["elapsed_seconds"] / 3600,
                "peak_gpu_memory_gib": completion["peak_gpu_memory_gib"], **execution_details(run)})
    pd.DataFrame(run_details).to_csv(final / "run_details.csv", index=False)
    val_table = pd.DataFrame(validation_rows)
    val_table.to_csv(final / "validation_per_seed.csv", index=False)
    val_summary = aggregate(validation_rows, SCALARS)
    val_summary.to_csv(final / "validation_summary.csv", index=False)
    paired = val_table[val_table.model == "roberta_mean"].merge(
        val_table[val_table.model == "roberta_first"], on="seed", suffixes=("_mean", "_first"))
    paired["macro_f1_mean_minus_first"] = paired.macro_f1_mean - paired.macro_f1_first
    paired["mae_mean_minus_first"] = paired.mae_mean - paired.mae_first
    paired.to_csv(final / "ablation_paired.csv", index=False)
    pooled = val_summary[val_summary.model.isin(["roberta_mean", "roberta_first"])]
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(pooled.model, pooled.macro_f1_mean, yerr=pooled.macro_f1_std, capsize=5)
    axis.set(ylabel="Validation Macro F1", title="Pooling ablation: paired seeds, mean +/- sample SD")
    save_figure(fig, figures / "pooling_ablation.png")
    rows, per_class, model_confusions = [], [], {}
    for model in ("baseline", "roberta_mean"):
        model_confusions[model] = np.zeros((5, 5), dtype=np.int64)
        for seed in config["seeds"]:
            name = f"{model}_seed{seed}"
            metrics = read_json(final / "metrics" / f"{name}.json")["metrics"]
            pred = pd.read_csv(final / "predictions" / f"{name}.csv")
            check = classification_metrics(pred.true_stars.to_numpy() - 1, pred.predicted_stars.to_numpy() - 1)
            for metric in SCALARS:
                if not np.isclose(check[metric], metrics[metric]):
                    raise ValueError(f"Saved metrics do not match predictions: {name}")
            rows.append({"model": model, "seed": seed, **{key: metrics[key] for key in SCALARS}})
            for rating, f1 in enumerate(metrics["per_class_f1"], 1):
                per_class.append({"model": model, "seed": seed, "stars": rating, "f1": f1})
            model_confusions[model] += np.asarray(metrics["confusion_matrix"])
            confusion_plot(metrics["confusion_matrix"], f"{name}: final test", figures / f"{name}_confusion.png")
        confusion_plot(model_confusions[model], f"{model}: mean normalized over seeds", figures / f"{model}_confusion_mean.png")
    pd.DataFrame(rows).to_csv(final / "test_per_seed.csv", index=False)
    pd.DataFrame(per_class).to_csv(final / "per_class_per_seed.csv", index=False)
    pd.DataFrame(per_class).groupby(["model", "stars"]).f1.agg(["mean", "std"]).reset_index().to_csv(
        final / "per_class_summary.csv", index=False)
    summary = aggregate(rows, SCALARS)
    summary.to_csv(final / "test_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, metric in zip(axes, ("accuracy", "macro_f1", "mae")):
        axis.bar(summary.model, summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"], capsize=5)
        axis.set(ylabel=metric, title="Lower is better" if metric == "mae" else "Higher is better")
    fig.suptitle("Final test: mean +/- sample SD across seeds")
    save_figure(fig, figures / "model_comparison.png")
    example = qualitative_examples(config, final, log)
    token_importance(config, prepared, runs, final, example, log)
    report_draft(config, prepared, runs, final, summary, val_summary)
    log.info("Analysis saved: tables, training curves, confusion matrices, ordinal errors, qualitative examples and report draft")
