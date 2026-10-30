"""Reject stale or changed data/tokenizer files using a portable semantic manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2

from data import read_rows


REVISION = '96df5e686bee6baa90b8bee7c28b81fa3fa6223d'


def rows_fingerprint(rows):
    hasher = hashlib.sha256()
    for row in rows:
        assert set(row) == {'id', 'article', 'highlights'}
        assert all(isinstance(row[key], str) and row[key].strip() for key in row)
        hasher.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':')).encode('utf-8') + b'\n')
    assert len({row['id'] for row in rows}) == len(rows), 'Duplicate dataset IDs'
    return hasher.hexdigest()


def tokenizer_fingerprint(path):
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(Path(path).read_bytes())
    tokenizer = spm.SentencePieceProcessor(model_file=str(path))
    assert proto.trainer_spec.model_type == 2, 'Expected BPE tokenizer'
    assert tokenizer.vocab_size() == 4096, 'Expected vocabulary size 4096'
    assert [tokenizer.pad_id(), tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()] == [0,1,2,3]
    # Trainer file paths are intentionally excluded so this works across machines.
    content = dict(pieces=[(piece.piece, piece.score, piece.type) for piece in proto.pieces],
                   normalizer=proto.normalizer_spec.SerializeToString().hex())
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode('utf-8')).hexdigest()


def verify(work, require_tokenizer=True, splits=('train', 'validation', 'test')):
    expected = json.loads(Path(__file__).with_name('input_manifest.json').read_text())
    ids = []
    for split in splits:
        rows = read_rows(Path(work) / f'{split}.jsonl')
        declared = expected['splits'][split]
        if len(rows) != declared['count'] or rows_fingerprint(rows) != declared['semantic_sha256']:
            raise ValueError(f'{split} does not match the declared CNN/Daily Mail subset')
        ids.append({row['id'] for row in rows})
    for i, left in enumerate(ids):
        assert all(not (left & right) for right in ids[i+1:]), 'Split IDs overlap'
    if require_tokenizer:
        if tokenizer_fingerprint(Path(work) / 'tokenizer.model') != expected['tokenizer_semantic_sha256']:
            raise ValueError('Tokenizer differs from the declared training-only BPE vocabulary')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='../work')
    args = parser.parse_args()
    verify(args.work)
    print('Dataset counts, semantic hashes, split IDs and tokenizer configuration: PASS')
