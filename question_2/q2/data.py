"""Training-only vocabulary, strict title recovery, GloVe and two tokenization paths."""
import argparse
from collections import Counter
import html
import json
import re
import shutil
import time
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .common import digest, environment, file_hash, load_config, logger, read_json, write_json

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/1db2080b2d94def6e5b0386a523102f9d8849e9d/glove.6B.zip"
GLOVE_SHA256 = "6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837"
TOKEN_PATTERN = re.compile(r"\w+(?:['\u2019]\w+)*|[^\w\s]", re.UNICODE)


def classic_tokens(text):
    # Keep negation, punctuation and numbers; cleaning is only for the GRU path.
    text = html.unescape(str(text)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return TOKEN_PATTERN.findall(text)


def join_title_body(setfit_rows, original_rows, expected_size):
    """Recover titles only when ID, body and label all agree; preserve SetFit order."""
    if len(setfit_rows) != expected_size or len(original_rows) != expected_size:
        raise ValueError(f"Wrong split size: expected {expected_size}, got {len(setfit_rows)}, {len(original_rows)}")
    original = {r["review_id"]: r for r in original_rows}
    if len(original) != expected_size:
        raise ValueError("Duplicate IDs in title source")
    output, seen = [], set()
    for row in setfit_rows:
        rid = row["id"]
        if rid in seen or rid not in original:
            raise ValueError(f"Duplicate or missing review ID: {rid}")
        seen.add(rid)
        source = original[rid]
        if source["language"] != "en" or source["review_body"] != row["text"]:
            raise ValueError(f"Review body/language mismatch for {rid}")
        label = int(row["label"])
        if label not in range(5) or int(source["stars"]) != label + 1:
            raise ValueError(f"Label mismatch for {rid}")
        for field in ("review_title", "review_body"):
            if source[field] is not None and not isinstance(source[field], str):
                raise ValueError(f"Invalid {field} type for {rid}")
        # A few original records have a null title. Preserve the record and document
        # that missing field instead of inventing a title or changing split sizes.
        title, body = (source["review_title"] or "").strip(), (source["review_body"] or "").strip()
        text = f"{title}\n{body}".strip()
        if not text:
            raise ValueError(f"Empty review {rid}")
        output.append({"id": rid, "title": title, "body": body, "text": text, "label": label,
                       "product_id": source.get("product_id"), "reviewer_id": source.get("reviewer_id"),
                       "product_category": source.get("product_category")})
    counts = Counter(r["label"] for r in output)
    if set(counts) != set(range(5)):
        raise ValueError("Missing rating classes")
    return pd.DataFrame(output)


def load_split(config, split, log, source_dir=None):
    """Download only the requested split, not an eager DatasetDict containing test."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    d = config["dataset"]
    log.info("Loading %s split; pinned revisions; recovering titles by exact review ID", split)
    if source_dir:
        source_dir = Path(source_dir)
        raw_manifest = read_json(source_dir / "raw_manifest.json")
        if raw_manifest["dataset"] != d:
            raise ValueError("Attached raw dataset versions differ from config")
        paths = [source_dir / "setfit" / f"{split}.jsonl", source_dir / "original" / "en" / f"{split}.jsonl.gz"]
        for path in paths:
            if file_hash(path) != raw_manifest["files"][path.relative_to(source_dir).as_posix()]:
                raise ValueError("Attached raw dataset file changed")
    else:
        paths = [hf_hub_download(repo_id=d["repo"], repo_type="dataset", revision=d["revision"],
                                  filename=f"{split}.jsonl"),
                 hf_hub_download(repo_id=d["title_repo"], repo_type="dataset", revision=d["title_revision"],
                                  filename=f"en/{split}.jsonl.gz")]
    rows = [load_dataset("json", data_files={split: path}, split=split) for path in paths]
    frame = join_title_body(rows[0], rows[1], d["expected_sizes"][split])
    log.info("%s: %d records; class counts=%s; titles=%d", split, len(frame),
             frame.label.value_counts().sort_index().to_dict(), int(frame.title.ne("").sum()))
    return frame


def build_vocab(texts, config):
    counter = Counter()
    for text in tqdm(texts, desc="Training-only vocabulary"):
        counter.update(classic_tokens(text))
    words = sorted((w for w, n in counter.items() if n >= config["min_frequency"]),
                   key=lambda w: (-counter[w], w))[:config["max_vocab"] - 2]
    return {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def encode_classic(texts, vocab, max_length):
    ids = np.zeros((len(texts), max_length), dtype=np.int32)
    lengths = np.zeros(len(texts), dtype=np.int32)
    token_count = unknown = truncated = 0
    raw_lengths = []
    for i, text in enumerate(tqdm(texts, desc="Classic tokenization")):
        tokens = classic_tokens(text)
        raw_lengths.append(len(tokens))
        truncated += len(tokens) > max_length
        tokens = tokens[:max_length] or ["<unk>"]
        values = [vocab.get(t, 1) for t in tokens]
        ids[i, :len(values)], lengths[i] = values, len(values)
        token_count += len(values)
        unknown += values.count(1)
    stats = {"unknown_token_rate": unknown / max(1, token_count),
             "truncation_rate": truncated / len(texts),
             "length_percentiles": np.percentile(raw_lengths, [50, 90, 95, 100]).tolist()}
    return ids, lengths, stats


def download_file(url, path, log):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as response:
                response.raise_for_status()
                resume = offset > 0 and response.status_code == 206
                if not resume:
                    offset = 0
                total = int(response.headers.get("Content-Length", 0)) + offset
                with open(temporary, "ab" if resume else "wb") as handle, tqdm(
                        total=total or None, initial=offset, unit="B", unit_scale=True, desc=path.name) as bar:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            bar.update(len(chunk))
            temporary.replace(path)
            log.info("Downloaded %s; SHA256=%s", path.name, file_hash(path))
            return path
        except requests.RequestException:
            if attempt == 2:
                raise
            log.warning("Download interrupted; retrying (%d/3)", attempt + 2)
            time.sleep(2)


def build_embeddings(vocab, dimension, output, glove_path, log):
    # Only OOV rows use fixed random initialization; pretrained rows are mandatory.
    generator = np.random.default_rng(2026)
    matrix = generator.normal(0, 0.05, (len(vocab), dimension)).astype(np.float32)
    matrix[0] = 0
    if glove_path:
        source = Path(glove_path)
    else:
        source = download_file(GLOVE_URL, output.parent / "downloads" / "glove.6B.zip", log)
    checksum = file_hash(source)
    if not glove_path and checksum != GLOVE_SHA256:
        source.unlink()
        raise ValueError("Downloaded GloVe archive checksum mismatch; removed corrupt download")
    archive = zipfile.ZipFile(source) if source.suffix == ".zip" else None
    stream = archive.open(f"glove.6B.{dimension}d.txt") if archive else open(source, "rb")
    matched = set()
    try:
        for raw in tqdm(stream, desc="Reading pretrained GloVe", total=400000):
            fields = raw.decode("utf-8").rstrip().split(" ")
            word = fields[0]
            if word not in vocab or word in ("<pad>", "<unk>"):
                continue
            vector = np.asarray(fields[1:], dtype=np.float32)
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid GloVe vector for {word}; expected {dimension} dimensions")
            matrix[vocab[word]] = vector
            matched.add(word)
    finally:
        stream.close()
        if archive:
            archive.close()
    coverage = len(matched) / max(1, len(vocab) - 2)
    if coverage < 0.25:
        raise ValueError(f"Pretrained embedding coverage too low ({coverage:.1%}); check supplied GloVe file")
    np.save(output / "embeddings.npy", matrix)
    stats = {"source": str(source) if glove_path else GLOVE_URL, "sha256": checksum,
             "dimension": dimension, "matched_words": len(matched), "vocabulary_coverage": coverage,
             "oov_initialization_seed": 2026}
    log.info("GloVe coverage: %.2f%% (%d/%d)", coverage * 100, len(matched), len(vocab) - 2)
    # Keep only the compact vocabulary-specific matrix in the prepared artifact.
    if not glove_path:
        source.unlink()
    return stats


def preparation_key(config):
    return digest({"dataset": config["dataset"], "baseline": {k: config["baseline"][k] for k in
                   ("max_length", "min_frequency", "max_vocab", "embedding_dim")},
                   "roberta": {k: config["roberta"][k] for k in ("model_name", "revision", "max_length")},
                   "preprocessing_version": 1, "implementation_sha256": file_hash(__file__)})


def prepare(config, output, glove_path=None, source_dir=None):
    from datasets import Dataset as HFDataset
    from huggingface_hub import snapshot_download
    from transformers import RobertaTokenizerFast
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = logger(output / "preparation.log")
    key = preparation_key(config)
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        if manifest["preparation_key"] != key:
            raise ValueError("Preparation config changed; use a new output directory")
        verify_prepared(output, config)
        log.info("Prepared data already complete and verified; skipping")
        return
    start = time.monotonic()
    train, validation = (load_split(config, split, log, source_dir) for split in ("train", "validation"))
    if set(train.id) & set(validation.id):
        raise ValueError("Review IDs overlap across training/validation")
    overlap = len(set(train.text) & set(validation.text))
    if overlap:
        log.warning("%d exact texts occur in both official splits; documented, official splits preserved", overlap)
    from .audit import audit_frames
    audit_frames({"train": train, "validation": validation}, config, output / "audit")
    vocab = build_vocab(train.text, config["baseline"])
    write_json(output / "vocab.json", vocab)
    stats = {"exact_train_validation_text_overlap": overlap}
    for name, frame in (("train", train), ("validation", validation)):
        frame.to_parquet(output / f"{name}.parquet", index=False)
        ids, lengths, stat = encode_classic(frame.text.tolist(), vocab, config["baseline"]["max_length"])
        np.savez_compressed(output / f"baseline_{name}.npz", input_ids=ids, lengths=lengths,
                            labels=frame.label.to_numpy(dtype=np.int64))
        stats[name] = {"rows": len(frame), "missing_titles": int(frame.title.eq("").sum()),
                       "missing_bodies": int(frame.body.eq("").sum()), "class_counts": frame.label.value_counts().sort_index().to_dict(), **stat}
    stats["vocabulary_size"] = len(vocab)
    stats["embeddings"] = build_embeddings(vocab, config["baseline"]["embedding_dim"], output, glove_path, log)
    pretrained = output / "pretrained"
    log.info("Downloading pinned RoBERTa weights/tokenizer on CPU; training will reuse these files")
    snapshot_download(config["roberta"]["model_name"], revision=config["roberta"]["revision"], local_dir=pretrained,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                      "vocab.json", "merges.txt", "special_tokens_map.json"])
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained, local_files_only=True)
    tokenizer.save_pretrained(output / "tokenizer")
    for name, frame in (("train", train), ("validation", validation)):
        ds = HFDataset.from_dict({"text": frame.text.tolist(), "labels": frame.label.tolist()})
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=config["roberta"]["max_length"], padding=False)
        encoded = ds.map(tokenize, batched=True, remove_columns=["text"], desc=f"RoBERTa {name}")
        target = output / f"roberta_{name}"
        # A previous interrupted preparation can have left an incomplete Arrow directory.
        if target.exists():
            resolved = target.resolve()
            if resolved.parent != output.resolve():
                raise ValueError("Unexpected cache directory")
            shutil.rmtree(resolved)
        encoded.save_to_disk(str(target))
        stats[name]["roberta_at_length_limit_rate"] = float(np.mean(
            [len(x) >= config["roberta"]["max_length"] for x in encoded["input_ids"]]))
    write_json(output / "data_statistics.json", stats)
    checksums = {str(p.relative_to(output)).replace("\\", "/"): file_hash(p)
                 for p in output.rglob("*") if p.is_file() and not p.name.endswith(".log")
                 and ".cache" not in p.parts and p.name != "manifest.json"}
    write_json(output / "manifest.json", {"preparation_key": key, "config": config, "files": checksums,
               "environment": environment(), "elapsed_seconds": time.monotonic() - start,
               "test_accessed": False, "title_recovery": "exact ID, body, language and stars validation"})
    log.info("PREPARATION COMPLETE in %.1f minutes. Full train/validation only; test untouched.",
             (time.monotonic() - start) / 60)


def verify_prepared(path, config):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest["preparation_key"] != preparation_key(config):
        raise ValueError("Prepared data does not match this configuration")
    for relative, checksum in manifest["files"].items():
        if file_hash(path / relative) != checksum:
            raise ValueError(f"Prepared file changed/corrupted: {relative}")
    return manifest


class ClassicDataset(Dataset):
    def __init__(self, arrays):
        self.ids = arrays["input_ids"]
        self.lengths = arrays["lengths"]
        self.labels = arrays["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": torch.tensor(self.ids[index], dtype=torch.long),
                "lengths": torch.tensor(self.lengths[index], dtype=torch.long),
                "labels": torch.tensor(self.labels[index], dtype=torch.long)}


def load_training_data(prepared, model):
    prepared = Path(prepared)
    if model == "baseline":
        return [ClassicDataset(np.load(prepared / f"baseline_{s}.npz")) for s in ("train", "validation")]
    from datasets import load_from_disk
    return [load_from_disk(str(prepared / f"roberta_{s}")) for s in ("train", "validation")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-data", help="Optional attached directory produced by download_data.py")
    parser.add_argument("--glove-path", help="Optional attached GloVe zip or matching .txt file")
    args = parser.parse_args()
    prepare(load_config(args.config), args.output, args.glove_path, args.raw_data)


if __name__ == "__main__":
    main()
