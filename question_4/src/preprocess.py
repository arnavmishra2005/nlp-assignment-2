import json

import torch

from src import config
from src.data import Vocab, clean_text, encode_story, load_story_splits, tokenize


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_story_splits()
    tokens_by_split = {}
    for name, stories in splits.items():
        cleaned = [clean_text(story) for story in stories]
        tokens_by_split[name] = [tokenize(story) for story in cleaned if story]

    vocab = Vocab.build(tokens_by_split["train"])
    vocab.save(config.ARTIFACT_DIR / "vocab.json")

    stats = {"vocab_size": len(vocab)}
    for name, token_lists in tokens_by_split.items():
        sequences = [encode_story(vocab, tokens) for tokens in token_lists]
        torch.save(sequences, config.DATA_DIR / f"{name}.pt")

        lengths = [len(seq) for seq in sequences]
        total_tokens = sum(lengths)
        unk_count = sum(seq.count(vocab.unk_id) for seq in sequences)
        truncated = sum(len(tokens) > config.MAX_LEN for tokens in token_lists)
        stats[name] = {
            "stories": len(sequences),
            "tokens": total_tokens,
            "avg_len": round(total_tokens / len(sequences), 1),
            "max_len": max(lengths),
            "unk_rate_percent": round(100 * unk_count / total_tokens, 2),
            "truncated_percent": round(100 * truncated / len(sequences), 1),
        }

    with open(config.ARTIFACT_DIR / "data_stats.json", "w") as file:
        json.dump(stats, file, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
