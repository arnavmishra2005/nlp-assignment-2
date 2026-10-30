import argparse
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn

from src import config
from src.data import Vocab, build_dataloader, load_processed
from src.models import StoryLM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(training):
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            tokens = (targets != criterion.ignore_index).sum().item()
            if training:
                optimizer.zero_grad()
                (loss / tokens).backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item()
            total_tokens += tokens
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "lstm"], required=True)
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    config.ARTIFACT_DIR.mkdir(exist_ok=True)

    vocab = Vocab.load(config.ARTIFACT_DIR / "vocab.json")
    loaders = {
        "train": build_dataloader(load_processed("train"), vocab.pad_id, shuffle=True),
        "val": build_dataloader(load_processed("val"), vocab.pad_id, shuffle=False),
        "test": build_dataloader(load_processed("test"), vocab.pad_id, shuffle=False),
    }

    model = StoryLM(
        len(vocab),
        args.model,
        config.EMBED_DIM,
        config.HIDDEN_DIM,
        config.NUM_LAYERS,
        config.DROPOUT,
        vocab.pad_id,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} device={device} parameters={parameter_count:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {
        "model": args.model,
        "seed": args.seed,
        "parameters": parameter_count,
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
    }
    checkpoint_path = config.CHECKPOINT_DIR / f"{args.model}.pt"
    best_val = float("inf")
    bad_epochs = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss = run_epoch(model, loaders["val"], criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(math.exp(val_loss))
        print(
            f"epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_ppl={math.exp(val_loss):.2f} elapsed={time.time() - start:.0f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                print("early stopping")
                break

    history["train_seconds"] = time.time() - start
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss = run_epoch(model, loaders["test"], criterion, device)
    history["best_val_loss"] = best_val
    history["best_val_perplexity"] = math.exp(best_val)
    history["test_loss"] = test_loss
    history["test_perplexity"] = math.exp(test_loss)
    print(f"best_val_ppl={math.exp(best_val):.2f} test_ppl={math.exp(test_loss):.2f}")

    with open(config.ARTIFACT_DIR / f"{args.model}_history.json", "w") as file:
        json.dump(history, file, indent=2)


if __name__ == "__main__":
    main()
