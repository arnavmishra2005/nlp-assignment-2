import argparse
import json
import random
from pathlib import Path

import torch

from src import config
from src.data import Vocab, clean_text, tokenize
from src.models import StoryLM

PROMPTS = [
    "One night, a student found an exam paper with tomorrow's answers.",
    "Mia opened the old box and discovered something nobody had seen before.",
    "The detective solved the murder before it happened.",
    "One morning, everyone in the village forgot the previous day.",
    "A letter arrived from exactly one hundred years in the future.",
    "A shy robot wanted to make one human friend.",
    "The little fox had to solve a problem before sunset."
]

def load_model(name, device, vocab):
    model = StoryLM(len(vocab), name, config.EMBED_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT, vocab.pad_id).to(device)
    state = torch.load(config.CHECKPOINT_DIR / f"{name}.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def choose_token(logits, strategy, temperature, top_k):
    logits = logits.clone()
    logits[config.PAD_TOKEN_ID if hasattr(config, "PAD_TOKEN_ID") else 0] = float("-inf")
    logits[2] = float("-inf")
    if strategy == "greedy":
        return int(logits.argmax().item())
    if strategy == "temperature":
        logits = logits / max(temperature, 1e-5)
    elif strategy == "top_k":
        k = min(top_k, logits.numel())
        values, indices = torch.topk(logits, k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered[indices] = values
        logits = filtered
    else:
        raise ValueError("Unknown strategy")
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

@torch.no_grad()
def generate(model, vocab, prompt, device, strategy="greedy", temperature=0.8, top_k=10, max_new_tokens=100):
    prompt_ids = [vocab.bos_id] + vocab.encode(tokenize(clean_text(prompt)))
    prompt_ids = prompt_ids[-(config.MAX_LEN - max_new_tokens if max_new_tokens < config.MAX_LEN else 1):]
    current = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = []
    for _ in range(max_new_tokens):
        logits, _ = model(current)
        next_id = choose_token(logits[0, -1], strategy, temperature, top_k)
        if next_id == vocab.eos_id:
            break
        output_ids.append(next_id)
        current = torch.cat([current, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if current.size(1) >= config.MAX_LEN:
            break
    continuation = " ".join(vocab.decode(output_ids))
    return prompt.rstrip() + (" " if continuation else "") + continuation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "lstm"], required=True)
    parser.add_argument("--strategy", choices=["greedy", "temperature", "top_k"], default="greedy")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--prompt_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(config.ARTIFACT_DIR / "vocab.json")
    model = load_model(args.model, device, vocab)
    prompts = PROMPTS if args.prompt_index < 0 else [PROMPTS[args.prompt_index]]
    results = []
    for prompt in prompts:
        output = generate(model, vocab, prompt, device, args.strategy, args.temperature, args.top_k, args.max_new_tokens)
        item = {"model": args.model, "strategy": args.strategy, "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed, "prompt": prompt, "output": output}
        results.append(item)
        print("\n" + "=" * 80 + "\n" + output)
    Path("results").mkdir(exist_ok=True)
    path = Path("results") / f"{args.model}_{args.strategy}_t{args.temperature}_k{args.top_k}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {path}")

if __name__ == "__main__":
    main()
