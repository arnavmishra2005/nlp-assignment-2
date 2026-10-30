"""Shared lowercased subword vocabulary, right shifting, and dynamic padding."""
import json
from pathlib import Path

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def train_tokenizer(rows, work, vocab_size=4096):
    text_path = Path(work) / 'tokenizer_training.txt'
    with text_path.open('w', encoding='utf-8') as stream:
        for row in rows:  # Only the training split is passed here.
            for field in ('article', 'highlights'):
                stream.write(row[field].lower().replace('\n', ' ') + '\n')
    spm.SentencePieceTrainer.train(
        input=str(text_path), model_prefix=str(Path(work) / 'tokenizer'),
        vocab_size=vocab_size, model_type='bpe', character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, num_threads=1,
        shuffle_input_sentence=False, hard_vocab_limit=False,
        max_sentence_length=100000)


class NewsDataset(Dataset):
    def __init__(self, rows, tokenizer, source_limit=400, target_limit=128):
        self.items = []
        for row in rows:
            source = tokenizer.encode(row['article'].lower())[:source_limit]
            summary = tokenizer.encode(row['highlights'].lower())[:target_limit - 2]
            self.items.append((torch.tensor(source or [tokenizer.unk_id()]),
                               torch.tensor([tokenizer.bos_id()] + summary + [tokenizer.eos_id()])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items):
    source, summary = zip(*items)
    source = pad_sequence(source, batch_first=True, padding_value=0)
    summary = pad_sequence(summary, batch_first=True, padding_value=0)
    return source, summary[:, :-1], summary[:, 1:]
