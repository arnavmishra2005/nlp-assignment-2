"""Optional CPU download of pinned training/validation raw files for later Add Input."""
import argparse
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
from q2.common import file_hash, load_config, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "validation"):
        for prefix, repo, revision, filename in (
            ("setfit", config["dataset"]["repo"], config["dataset"]["revision"], f"{split}.jsonl"),
            ("original", config["dataset"]["title_repo"], config["dataset"]["title_revision"], f"en/{split}.jsonl.gz")):
            source = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
            target = output / prefix / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files[target.relative_to(output).as_posix()] = file_hash(target)
            print("Downloaded", target, flush=True)
    write_json(output / "raw_manifest.json", {"dataset": config["dataset"], "files": files, "test_accessed": False})
    print("RAW_TRAIN_VALIDATION_DOWNLOAD_COMPLETE. Test deliberately excluded.", flush=True)
