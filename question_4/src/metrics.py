import json
import math
import random
import re
from pathlib import Path

import torch
from torch import nn

from src import config
from src.data import Vocab, build_dataloader, load_processed
from src.generate import PROMPTS, generate, load_model

def repetition_rate(text, n=3):
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return 1.0 - len(set(grams)) / len(grams)

def distinct_n(texts, n):
    grams = []
    for text in texts:
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
        grams.extend(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))
    return len(set(grams)) / max(1, len(grams))

def evaluate_loss(model, loader, device, pad_id):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    loss_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss_sum += criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
            token_count += (targets != pad_id).sum().item()
    loss = loss_sum / max(1, token_count)
    return loss, math.exp(min(loss, 20))

def main():
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(config.ARTIFACT_DIR / "vocab.json")
    test_loader = build_dataloader(load_processed("test"), vocab.pad_id, shuffle=False)
    results = []
    for name in ["rnn", "lstm"]:
        model = load_model(name, device, vocab)
        test_loss, test_ppl = evaluate_loss(model, test_loader, device, vocab.pad_id)
        outputs = []
        for prompt in PROMPTS:
            for strategy, temperature, top_k in [
                ("greedy", 1.0, 10),
                ("temperature", 0.5, 10),
                ("temperature", 0.8, 10),
                ("temperature", 1.2, 10),
                ("top_k", 1.0, 10)
            ]:
                outputs.append(generate(model, vocab, prompt, device, strategy, temperature, top_k, 80))
        lengths = [len(re.findall(r"\b\w+(?:'\w+)?\b", output)) for output in outputs]
        row = {
            "model": name,
            "test_loss": test_loss,
            "test_perplexity": test_ppl,
            "distinct_1": distinct_n(outputs, 1),
            "distinct_2": distinct_n(outputs, 2),
            "mean_repetition_rate_3gram": sum(repetition_rate(x) for x in outputs) / len(outputs),
            "average_generated_words_including_prompt": sum(lengths) / len(lengths),
            "sample_count": len(outputs),
            "premature_end_count_under_20_words": sum(length < 20 for length in lengths),
            "overlong_count_at_least_180_words": sum(length >= 180 for length in lengths),
            "repetition_definition": "1 - unique 3-gram count / total 3-gram count, averaged per output"
        }
        history_path = config.ARTIFACT_DIR / f"{name}_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            row["best_validation_perplexity"] = history.get("best_val_perplexity")
            row["parameters"] = history.get("parameters")
            row["train_seconds"] = history.get("train_seconds")
        results.append(row)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("Saved artifacts/metrics.json")

if __name__ == "__main__":
    main()
