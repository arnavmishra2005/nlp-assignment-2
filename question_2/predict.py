"""Optional CLI inference from your own trusted trained checkpoint. No demo required."""
import argparse
import json
from pathlib import Path
import torch
from q2.common import digest, file_hash, read_json, write_json
from q2.validation import verify_completed_run
from q2.data import encode_classic
from q2.models import load_trained


def predict(run, prepared, text, device="auto"):
    if not isinstance(text, str) or not text.strip() or len(text) > 20000:
        raise ValueError("Provide a nonempty title + review body of at most 20,000 characters")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    verify_completed_run(run)
    model, metadata = load_trained(run, device)
    prepared = Path(prepared)
    manifest = read_json(prepared / "manifest.json")
    if digest(manifest) != metadata["prepared_manifest"]:
        raise ValueError("Inference preprocessing does not match trained model")
    for relative, checksum in manifest["files"].items():
        if relative == "vocab.json" or relative.startswith("tokenizer/"):
            if file_hash(prepared / relative) != checksum:
                raise ValueError("Inference vocabulary/tokenizer changed")
    if metadata["model"] == "baseline":
        import numpy as np
        ids, lengths, _ = encode_classic([text], read_json(prepared / "vocab.json"), metadata["config"]["baseline"]["max_length"])
        inputs = {"input_ids": torch.tensor(ids, dtype=torch.long, device=device),
                  "lengths": torch.tensor(lengths, dtype=torch.long, device=device)}
    else:
        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained(prepared / "tokenizer", local_files_only=True)
        inputs = {k: v.to(device) for k, v in tokenizer(text, truncation=True,
            max_length=metadata["config"]["roberta"]["max_length"], return_tensors="pt").items()}
    with torch.inference_mode():
        probability = model(**inputs).float().softmax(-1)[0].cpu().tolist()
    return {"predicted_stars": int(max(range(5), key=probability.__getitem__) + 1),
            "probabilities_1_to_5": probability, "model": metadata["model"], "seed": metadata["seed"],
            "device": str(device), "note": "Assignment classifier; probabilities are not established to be calibrated"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--prepared", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--text-file")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
    result = predict(args.run, args.prepared, text, args.device)
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)
