"""Narrow compatibility for completed pre-storage-change prepare/smoke evidence.

Only known completed prerequisite sources/phases are accepted. Original data,
models and single-GPU trainer must match exactly; the preflight message has a
separately pinned logging-only replacement. The completed original mean-pooling
seed42 run is retained at the user's request, with its single-GPU provenance.
The original source fingerprint is retained and reported, never relabeled.
"""
from pathlib import Path
from .common import file_hash, source_hash

LEGACY = {'legacy_source_hash': '9e9a6d4bc8db0a17c20f732ec8f1a17221c7d1c9621c04ff1497d5a5a76c92ee', 'core_files': {'common.py': '48d124b753a824e27628446fc0e040cecd9cd151c4b1506198c045e46dcedd4f', 'data.py': '08c77fb646e307debfced3cd79e2b9c6dabe13775d8be06907b35fc0cf9ae3cb', 'audit.py': '8dd5109296f5dc9779ad99039d1fc9cb19ec018c71605768dc5a03b39e14e640', 'models.py': '734b4a35056e4b36182b44e9f8f8fd0ef70683ec3c2fe554203778535f8b316a', 'train.py': 'd4adfa4af2adc432d7289bff214dfa83c6c0fcdfe6849048352f3b51a3a9118d', 'metrics.py': 'b36166047b565f94ded19db764d79ad574d2f9fc97e8e915c19d72676e2f7b4b', 'validation.py': '88970e18c5b32534c0d236b791409e8693f45024f83a759468be92ee264ff90c', 'smoke.py': 'c5042a0c71267d4f73e28715f00fd36e62ede9707a7cee88916a77102fc28a72', 'preflight.py': 'a013e892d3e9d95c5f488fc38c49df37515b4e036738ed8c869525270908e8be'}}


def compatible_prerequisite(manifest, expected_source):
    phase, original = manifest.get('phase'), manifest.get('source_hash')
    accepted = (original == LEGACY['legacy_source_hash'] and phase in ('prepare', 'smoke', 'baseline_seed42')) or (
        original == '5c284bdea9b164be9eb13cfb897f1f8e56cb8ca19bb3d0342ca80362a9d42144'
        and phase in ('prepare', 'smoke', 'baseline_seed42', 'baseline_seed123', 'roberta_mean_seed42'))
    if not accepted:
        return False
    if expected_source != source_hash():
        return False
    root = Path(__file__).parent
    expected_core = {**LEGACY['core_files'], 'preflight.py': PREFLIGHT_LOGGING_HASH}
    return all(file_hash(root / name) == checksum for name, checksum in expected_core.items())


PREFLIGHT_LOGGING_HASH = 'cd80020fb3cfbbb2f663e0ee5c8612872d1c9240eadcc8053725ecf59f24fdb0'
