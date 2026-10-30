import os, sys, json, time, random
import torch
import torch.nn.functional as F
from data import prepare, encode, pad, PAD
from models import Translator

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def make_batches(pairs, batch_size, shuffle):
    order = list(range(len(pairs)))
    if shuffle:
        random.shuffle(order)
    # sort by length inside chunks so a batch holds similar lengths (less padding, faster on cpu)
    chunk = batch_size * 50
    batches = []
    for i in range(0, len(order), chunk):
        part = sorted(order[i:i + chunk], key=lambda j: len(pairs[j][0]))
        batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def to_tensors(batch_pairs, data):
    src, src_len = pad([encode(e, data["en_stoi"]) for e, _ in batch_pairs])
    tgt, _ = pad([encode(h, data["hi_stoi"]) for _, h in batch_pairs])
    return torch.tensor(src), torch.tensor(src_len), torch.tensor(tgt)


def run_epoch(model, pairs, data, optimizer=None, batch_size=64, tag=""):
    training = optimizer is not None
    model.train(training)
    batches = make_batches(pairs, batch_size, shuffle=training)
    total_loss, total_correct, total_words = 0.0, 0, 0
    start = time.time()
    for n, idx in enumerate(batches, 1):
        src, src_len, tgt = to_tensors([pairs[j] for j in idx], data)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.set_grad_enabled(training):
            logits = model(src, src_len, tgt_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD)
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        real = tgt_out != PAD
        total_loss += loss.item() * real.sum().item()
        total_correct += ((logits.argmax(-1) == tgt_out) & real).sum().item()
        total_words += real.sum().item()
        if training and n % 100 == 0:
            elapsed = time.time() - start
            print(f"{tag} step {n}/{len(batches)}  loss {total_loss / total_words:.3f}  "
                  f"elapsed {elapsed:.0f}s  eta {elapsed / n * (len(batches) - n):.0f}s", flush=True)
    return total_loss / total_words, total_correct / total_words


def train(name, use_attention, epochs=8, threads=6, lr=2e-3, batch_size=128):
    torch.set_num_threads(threads)
    random.seed(0); torch.manual_seed(0)
    data = prepare(verbose=False)
    model = Translator(len(data["en_itos"]), len(data["hi_itos"]), use_attention=use_attention)
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} parameters, {epochs} epochs, {threads} threads", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], epoch_time=[])
    best = 1e9
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, data["train"], data, optimizer, batch_size, tag=f"[{name} e{epoch}]")
        va_loss, va_acc = run_epoch(model, data["dev"], data, batch_size=batch_size)
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc, time.time() - t0)):
            history[k].append(v)
        saved = ""
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), os.path.join(RESULTS, f"{name}.pt"))
            saved = "  (best so far, saved)"
        print(f"[{name}] epoch {epoch}/{epochs}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}  time {time.time() - t0:.0f}s{saved}", flush=True)
        json.dump(history, open(os.path.join(RESULTS, f"{name}_history.json"), "w"))


if __name__ == "__main__":
    which = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    train(which, use_attention=(which == "attention"), epochs=epochs, threads=threads)
