"""Create a blinded human-rating sheet and a separate answer key.

Run from the project root: python -m src.human_eval
Do not share results/human_eval_key.csv with raters before ratings are complete.
"""
import csv
import json
import random
from pathlib import Path

def main():
    source = Path("results/plot_twist_challenge.json")
    if not source.exists():
        raise FileNotFoundError("Run python -m src.challenge first.")
    records = json.loads(source.read_text(encoding="utf-8"))
    groups = {}
    for record in records:
        if record.get("sample_index", 1) == 1:  # avoid duplicate deterministic greedy samples
            groups.setdefault((record["model"], record["strategy"]), []).append(record)
    rng = random.Random(42)
    selected = []
    for model in ("rnn", "lstm"):
        for strategy in ("greedy", "temperature", "top_k"):
            group = groups[(model, strategy)].copy()
            rng.shuffle(group)
            selected.extend(group[:5])
    rng.shuffle(selected)
    blind, key = [], []
    for i, record in enumerate(selected, 1):
        sample_id = f"S{i:02d}"
        prompt, output = record["prompt"], record["output"]
        continuation = output[len(prompt):].strip() if output.startswith(prompt) else output
        blind.append({"sample_id": sample_id, "prompt": prompt, "continuation": continuation,
                      "coherence_1_5": "", "grammar_1_5": "", "relevance_1_5": "",
                      "creativity_surprise_1_5": "", "rater_id": "", "notes": ""})
        key.append({"sample_id": sample_id, "model": record["model"], "strategy": record["strategy"],
                    "temperature": record.get("temperature"), "top_k": record.get("top_k"),
                    "sample_index": record.get("sample_index"), "prompt": prompt, "raw_output": output})
    Path("results").mkdir(exist_ok=True)
    with open("results/human_eval_blinded.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(blind[0])); w.writeheader(); w.writerows(blind)
    with open("results/human_eval_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    fields = ["sample_id", "rater_id", "prompt", "continuation", "coherence_1_5", "grammar_1_5",
              "relevance_1_5", "creativity_surprise_1_5", "notes"]
    with open("results/human_eval_3_rater_template.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in blind:
            for rater in ("R1", "R2", "R3"):
                w.writerow({"sample_id": row["sample_id"], "rater_id": rater, "prompt": row["prompt"],
                            "continuation": row["continuation"], "coherence_1_5": "", "grammar_1_5": "",
                            "relevance_1_5": "", "creativity_surprise_1_5": "", "notes": ""})
    print(f"Created blinded rating forms for {len(blind)} samples plus a separate answer key.")
    print("Collect genuine ratings before reporting human-evaluation results.")

if __name__ == "__main__":
    main()
