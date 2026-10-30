import csv
import json
import random
from pathlib import Path

def main():
    source = Path("results/plot_twist_challenge.json")
    if not source.exists():
        raise FileNotFoundError("Run python -m src.challenge first.")
    samples = json.loads(source.read_text(encoding="utf-8"))
    random.Random(42).shuffle(samples)
    selected = samples[:30]
    Path("results").mkdir(exist_ok=True)
    with open("results/human_eval_blinded.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "story", "coherence_1_5", "grammar_1_5", "relevance_1_5", "creativity_surprise_1_5", "rater_id", "notes"])
        for index, sample in enumerate(selected, 1):
            writer.writerow([f"S{index:02d}", sample["output"], "", "", "", "", "", ""])
    print(f"Created results/human_eval_blinded.csv with {len(selected)} samples.")
    print("Use at least 3 raters. Keep the model/strategy mapping separately from the blinded sheet.")

if __name__ == "__main__":
    main()
