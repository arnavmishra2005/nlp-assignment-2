"""Download CNN/Daily Mail 3.0.0 into a working directory, never the submission."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from verify_inputs import REVISION, rows_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='../work')
    parser.add_argument('--train-size', type=int, default=2048)
    parser.add_argument('--validation-size', type=int, default=1024)
    parser.add_argument('--verify-source', action='store_true',
                        help='Compare existing files with a fresh pinned repository stream')
    args = parser.parse_args()
    root = Path(args.work)
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(__file__).with_name('input_manifest.json').read_text())
    # The canonical repository is the current name of cnn_dailymail.
    for split, count in [('train', args.train_size),
                         ('validation', args.validation_size), ('test', 11490)]:
        path = root / f'{split}.jsonl'
        if count != manifest['splits'][split]['count']:
            raise ValueError('This reproduction manifest requires train=2048, validation=1024, test=11490')
        if path.exists():
            with path.open(encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream]
            expected = manifest['splits'][split]
            if len(rows) != count or rows_fingerprint(rows) != expected['semantic_sha256']:
                raise ValueError(f'Reused {split} file differs from the declared subset')
            print('Verified existing:', path, flush=True)
            if not args.verify_source:
                continue
        dataset = load_dataset('abisee/cnn_dailymail', '3.0.0', split=split,
                               revision=REVISION, streaming=True, cache_dir=str(root / 'hf_cache'))
        actual = 0
        temporary = path.with_suffix('.partial')
        with temporary.open('w', encoding='utf-8') as stream:
            for item in dataset.take(count):
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                actual += 1
        if actual != count:
            raise ValueError(f'{split}: expected {count}, got {actual}')
        with temporary.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        if rows_fingerprint(rows) != manifest['splits'][split]['semantic_sha256']:
            raise ValueError(f'Downloaded {split} differs from the declared pinned dataset')
        if args.verify_source and path.exists():
            temporary.unlink()
            print(f'{split}: remote pinned source matches the local subset', flush=True)
            continue
        temporary.replace(path)
        print(f'{split}: {actual} pairs saved', flush=True)


if __name__ == '__main__':
    main()
