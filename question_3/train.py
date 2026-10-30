"""Train with teacher forcing; select checkpoints using validation loss only."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import NewsDataset, collate, read_rows, train_tokenizer
from model import Transformer
from verify_inputs import verify


def epoch_loss(model, loader, device, optimizer=None, step=0, warmup=200, lr_scale=1.0):
    model.train(optimizer is not None)
    total_loss = total_tokens = 0
    for source, decoder_input, target in loader:
        source, decoder_input, target = [x.to(device) for x in (source, decoder_input, target)]
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(source, decoder_input)
            loss_sum = F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                       ignore_index=0, reduction='sum')
            tokens = target.ne(0).sum()
            if optimizer is not None:
                step += 1
                rate = lr_scale * model.width ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
                for group in optimizer.param_groups:
                    group['lr'] = rate
                optimizer.zero_grad(set_to_none=True)
                (loss_sum / tokens).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss_sum.item()
        total_tokens += tokens.item()
    return total_loss / total_tokens, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='../work')
    parser.add_argument('--output', default='results')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr-scale', type=float, default=0.1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    work, output = Path(args.work), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training, validation = read_rows(work / 'train.jsonl'), read_rows(work / 'validation.jsonl')
    if not (work / 'tokenizer.model').exists():
        train_tokenizer(training, work)
    verify(work, splits=('train', 'validation'))
    tokenizer = spm.SentencePieceProcessor(model_file=str(work / 'tokenizer.model'))
    config = dict(vocab_size=len(tokenizer), width=64, heads=4, layers=2,
                  hidden=256, dropout=0.1, pad_id=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    train_loader = DataLoader(NewsDataset(training, tokenizer), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(NewsDataset(validation, tokenizer), batch_size=args.batch_size,
                            collate_fn=collate)
    history, best, stale, step = [], float('inf'), 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, step = epoch_loss(model, train_loader, device, optimizer, step,
                                     lr_scale=args.lr_scale)
        val_loss, _ = epoch_loss(model, val_loader, device)
        history.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                            learning_rate=optimizer.param_groups[0]['lr']))
        print(f'Epoch {epoch}: train={train_loss:.4f}, validation={val_loss:.4f}', flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                            validation_loss=best), work / 'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(work / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    with (output / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    metadata = dict(config=config, parameters=sum(p.numel() for p in model.parameters()),
                    train_pairs=len(training), validation_pairs=len(validation),
                    source_limit=400, summary_limit=128, generation_limit=128,
                    seed=42, batch_size=args.batch_size, warmup_steps=200,
                    lr_scale=args.lr_scale,
                    patience=args.patience, best_epoch=checkpoint['epoch'],
                    best_validation_loss=best, training_seconds=time.perf_counter() - started,
                    device=str(device), threads=args.threads, torch_version=torch.__version__,
                    tokenizer='SentencePiece BPE, lowercased training split only')
    (output / 'training.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
