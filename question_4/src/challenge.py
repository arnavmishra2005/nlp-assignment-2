import json
from pathlib import Path

import torch

from src import config
from src.data import Vocab
from src.generate import load_model, generate

TWIST_PROMPTS = [
    "A student found an exam paper containing tomorrow's answers.",
    "An old robot received a message from its future self.",
    "A detective solved a murder that had not happened yet.",
    "One morning, a whole village forgot the previous day.",
    "A letter arrived from one hundred years in the future.",
    "The new kid at school knew everyone's secrets but had never met them.",
    "Every night, the moon moved closer to the village."
]

def main():
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(config.ARTIFACT_DIR / "vocab.json")
    Path("results").mkdir(exist_ok=True)
    all_outputs = []
    for model_name in ["rnn", "lstm"]:
        model = load_model(model_name, device, vocab)
        for strategy, temperature, top_k in [("greedy", 1.0, 10), ("temperature", 0.8, 10), ("top_k", 1.0, 10)]:
            for prompt in TWIST_PROMPTS:
                for sample_index in range(1, 3):
                    output = generate(model, vocab, prompt, device, strategy, temperature, top_k, 80)
                    item = {"model": model_name, "strategy": strategy, "temperature": temperature, "top_k": top_k, "sample_index": sample_index, "prompt": prompt, "output": output}
                    all_outputs.append(item)
                    print(f"\n[{model_name} | {strategy} | sample {sample_index}]\n{output}\n")
    path = Path("results/plot_twist_challenge.json")
    path.write_text(json.dumps(all_outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(all_outputs)} raw outputs to {path}")

if __name__ == "__main__":
    main()
