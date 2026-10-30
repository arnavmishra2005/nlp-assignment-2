import json
import random
import re
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src import config

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]")


def load_story_splits():
    from datasets import load_dataset

    stream = load_dataset(config.DATASET_NAME, split="train", streaming=True)
    total = config.NUM_TRAIN + config.NUM_VAL + config.NUM_TEST
    stories = [row["text"] for row in stream.take(total)]
    random.Random(config.SEED).shuffle(stories)

    val_start = config.NUM_TRAIN
    test_start = config.NUM_TRAIN + config.NUM_VAL
    return {
        "train": stories[:val_start],
        "val": stories[val_start:test_start],
        "test": stories[test_start:total],
    }


def clean_text(text):
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {token: index for index, token in enumerate(itos)}
        self.pad_id = self.stoi[config.PAD_TOKEN]
        self.unk_id = self.stoi[config.UNK_TOKEN]
        self.bos_id = self.stoi[config.BOS_TOKEN]
        self.eos_id = self.stoi[config.EOS_TOKEN]

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, token_lists):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        room = config.MAX_VOCAB - len(config.SPECIAL_TOKENS)
        words = [word for word, count in counter.most_common() if count >= config.MIN_FREQ]
        return cls(config.SPECIAL_TOKENS + words[:room])

    def encode(self, tokens):
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def save(self, path):
        with open(path, "w") as file:
            json.dump(self.itos, file)

    @classmethod
    def load(cls, path):
        with open(path) as file:
            return cls(json.load(file))


def encode_story(vocab, tokens):
    truncated = len(tokens) > config.MAX_LEN
    ids = [vocab.bos_id] + vocab.encode(tokens[: config.MAX_LEN])
    if not truncated:
        ids.append(vocab.eos_id)
    return ids


class StoryDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        ids = torch.tensor(self.sequences[index], dtype=torch.long)
        return ids[:-1], ids[1:]


def make_collate_fn(pad_id):
    def collate(batch):
        inputs, targets = zip(*batch)
        lengths = torch.tensor([len(item) for item in inputs])
        inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        targets = pad_sequence(targets, batch_first=True, padding_value=pad_id)
        return inputs, targets, lengths

    return collate


def build_dataloader(sequences, pad_id, shuffle):
    return DataLoader(
        StoryDataset(sequences),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=make_collate_fn(pad_id),
    )


def load_processed(split):
    return torch.load(config.DATA_DIR / f"{split}.pt")
