"""Configuration, logging, provenance and atomic output helpers."""
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def read_json(path):
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def source_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def logger(output=None):
    log = logging.getLogger("q2")
    log.setLevel(logging.INFO)
    for old_handler in list(log.handlers):
        log.removeHandler(old_handler)
        old_handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    log.addHandler(handler)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(output, encoding="utf-8")
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def environment():
    import importlib.metadata
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "pandas", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "packages": packages,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count(), "cuda": torch.version.cuda,
            "source_hash": source_hash()}


def capture_dependencies(directory):
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (Path(directory) / "environment-freeze.txt").write_text(result.stdout, encoding="utf-8")


def load_config(path):
    config = read_json(path)
    if len(set(config["seeds"])) < 2:
        raise ValueError("At least two distinct seeds are required.")
    if config["baseline"]["num_layers"] != 2:
        raise ValueError("Baseline requires two GRU layers.")
    if not 0.06 <= config["roberta"]["warmup_ratio"] <= 0.10:
        raise ValueError("RoBERTa warmup must be 6-10%.")
    for key in ("baseline", "roberta"):
        c = config[key]
        if min(c["epochs"], c["batch_size"], c["accumulation_steps"], c["patience"]) < 1:
            raise ValueError("Epochs, batch size, accumulation and patience must be positive.")
    return config
