import os, json, logging
from collections import Counter
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sacrebleu
from data import prepare, EOS
from models import Translator
from train import to_tensors, make_batches, run_epoch, RESULTS

logging.getLogger("sacrebleu").setLevel(logging.ERROR)
BUCKETS = [("1-5", 1, 5), ("6-10", 6, 10), ("11-15", 11, 15), ("16+", 16, 999)]

for name in ["Nirmala UI", "Mangal", "Noto Sans Devanagari"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name
        break


def load_model(name, data):
    model = Translator(len(data["en_itos"]), len(data["hi_itos"]), use_attention=(name == "attention"))
    model.load_state_dict(torch.load(os.path.join(RESULTS, f"{name}.pt")))
    return model.eval()

def load_history(name):
    return json.load(open(os.path.join(RESULTS, f"{name}_history.json")))


def translate_all(model, pairs, data, batch_size=64, ban_unk=True):
    hyps, alphas = [None] * len(pairs), [None] * len(pairs)
    for idx in make_batches(pairs, batch_size, shuffle=False):
        src, src_len, _ = to_tensors([pairs[j] for j in idx], data)
        words, att = model.translate(src, src_len, ban_unk=ban_unk)
        for row, j in enumerate(idx):
            ids = words[row].tolist()
            n = ids.index(EOS) + 1 if EOS in ids else len(ids)
            hyps[j] = [data["hi_itos"][i] for i in ids[:n] if i != EOS]
            if att is not None:
                alphas[j] = att[row, :n, :src_len[row]].numpy()
    return hyps, alphas


def bleu(hyps, refs):
    if not hyps:
        return 0.0
    return sacrebleu.corpus_bleu([" ".join(h) for h in hyps], [[" ".join(r) for r in refs]], tokenize="none").score


def bucket_of(n):
    return next(name for name, lo, hi in BUCKETS if lo <= n <= hi)


def length_table(models, pairs, data, hyps):
    rows = []
    for name, lo, hi in BUCKETS:
        idx = [i for i, (e, _) in enumerate(pairs) if lo <= len(e) <= hi]
        row = {"bucket": name, "n": len(idx)}
        for m_name, model in models.items():
            sub = [pairs[i] for i in idx]
            row[f"{m_name}_bleu"] = bleu([hyps[m_name][i] for i in idx], [pairs[i][1] for i in idx])
            row[f"{m_name}_acc"] = run_epoch(model, sub, data)[1] if sub else 0.0
        rows.append(row)
    return rows


def first_last_recall(hyps, refs):
    # how many words from the first third / last third of the reference show up in the translation
    first, last, n_first, n_last = 0, 0, 0, 0
    for h, r in zip(hyps, refs):
        k = max(1, len(r) // 3)
        first += sum(w in h for w in r[:k]); n_first += k
        last += sum(w in h for w in r[-k:]); n_last += k
    return first / n_first, last / n_last


def classify(src, ref, hyp, alpha, data):
    labels = []
    if any(w not in data["hi_stoi"] for w in ref):
        labels.append("unknown/OOV word")
    missing = [w for w in ref if w not in hyp]
    extra = [w for w in hyp if w not in ref]
    if Counter(hyp) == Counter(ref) and hyp != ref:
        labels.append("incorrect word order")
    if len(hyp) < len(ref) - 1 and missing:
        labels.append("missing word")
    if len(hyp) > len(ref) + 1 and extra:
        labels.append("extra word")
    if any(c >= 3 for w, c in Counter(hyp).items() if w not in ".,।"):
        labels.append("repeated word")
    if extra and missing and abs(len(hyp) - len(ref)) <= 1:
        labels.append("incorrect word")
    if len(src) >= 16:
        labels.append("long-distance dependency")
    if alpha is not None and len(hyp) > 1:
        tokens = ["<SOS>"] + src + ["<EOS>"]
        peaks = [tokens[j] for j in alpha[:-1].argmax(axis=1)]
        if np.mean([t in ("<SOS>", "<EOS>", ".", ",", "?") for t in peaks]) > 0.5:
            labels.append("incorrect attention focus")
    return labels or ["other"]


def word_overlap(hyp, ref):
    common = sum((Counter(hyp) & Counter(ref)).values())
    return 2 * common / (len(hyp) + len(ref)) if hyp and ref else 0.0


def pick_error_cases(pairs, hyps_a, hyps_b, n=6):
    cands = [(i, word_overlap(hyps_b[i], pairs[i][1])) for i, (e, h) in enumerate(pairs)
             if 4 <= len(e) <= 20 and hyps_b[i] != h]
    cands.sort(key=lambda x: -x[1])
    chosen, seen = [], Counter()
    for i, score in cands:
        b = bucket_of(len(pairs[i][0]))
        if seen[b] < 2:
            chosen.append(i); seen[b] += 1
        if len(chosen) == n:
            break
    return chosen


def attention_stats(alphas):
    # per sentence: does the peak move left to right, how sharp is it
    moves, peaks, spread = [], [], []
    for a in alphas:
        if a is None or len(a) < 3:
            continue
        top = a[:-1].argmax(axis=1)
        moves.append(np.mean(top[1:] >= top[:-1]))
        peaks.append(a[:-1].max(axis=1).mean())
        spread.append(np.mean(a[:-1].max(axis=1) < 0.5))
    return np.mean(moves), np.mean(peaks), np.mean(spread)


def confidence_vs_quality(alphas, hyps, refs):
    pairs = [(a[:-1].max(axis=1).mean(), word_overlap(h, r)) for a, h, r in zip(alphas, hyps, refs) if a is not None and len(a) > 2]
    conf, quality = zip(*pairs)
    return np.corrcoef(conf, quality)[0, 1]


def plot_curves(histories, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"baseline": "tab:blue", "attention": "tab:red"}
    for name, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], "-o", color=colors[name], label=f"{name} train")
        axes[0].plot(ep, h["val_loss"], "--s", color=colors[name], label=f"{name} val")
        axes[1].plot(ep, h["train_acc"], "-o", color=colors[name], label=f"{name} train")
        axes[1].plot(ep, h["val_acc"], "--s", color=colors[name], label=f"{name} val")
    axes[0].set_title("Loss"); axes[1].set_title("Word accuracy (teacher forcing)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if path: plt.savefig(path, dpi=130)
    plt.show()


def plot_length(rows, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = [r["bucket"] for r in rows]
    for ax, key, title in [(axes[0], "bleu", "BLEU"), (axes[1], "acc", "Word accuracy")]:
        ax.plot(x, [r[f"baseline_{key}"] for r in rows], "-o", color="tab:blue", label="baseline")
        ax.plot(x, [r[f"attention_{key}"] for r in rows], "-o", color="tab:red", label="attention")
        ax.set_title(f"{title} by English sentence length"); ax.set_xlabel("English tokens")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if path: plt.savefig(path, dpi=130)
    plt.show()


def plot_heatmap(src_tokens, hyp_tokens, alpha, title="", path=None, ax=None):
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(max(5, 0.55 * alpha.shape[1]), max(3, 0.45 * alpha.shape[0])))
    ax.imshow(alpha, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(alpha.shape[1])); ax.set_xticklabels(["<SOS>"] + src_tokens + ["<EOS>"], rotation=60, ha="right")
    rows = hyp_tokens + ["<EOS>"] if alpha.shape[0] > len(hyp_tokens) else hyp_tokens
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xlabel("English"); ax.set_ylabel("generated Hindi"); ax.set_title(title, fontsize=9)
    if own:
        plt.tight_layout()
        if path: plt.savefig(path, dpi=130)
        plt.show()
