"""Create a source upload ZIP or a Q2 submission ZIP without raw data/checkpoints."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in {"__pycache__", "data", "artifacts", "dist", ".pytest_cache"} for part in relative.parts):
            continue
        if path.suffix in {".py", ".md", ".txt", ".json", ".ipynb"}:
            yield path, "question_2/" + relative.as_posix()


def package(output, results=None, runs=None, config=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in source_files():
            archive.write(Path(config) if config and name == "question_2/config.json" else path, name)
        if results:
            results = Path(results)
            if not (results / "EVALUATION_COMPLETE.json").exists():
                raise ValueError("Final evaluation is incomplete; cannot create a submission ZIP")
            for path in results.rglob("*"):
                if path.is_file() and "test_cache" not in path.relative_to(results).parts and path.suffix not in {".pt", ".tmp"}:
                    archive.write(path, "question_2/results/" + path.relative_to(results).as_posix())
        if runs:
            for path in Path(runs).rglob("*"):
                if path.is_file() and path.suffix in {".json", ".log", ".txt"}:
                    archive.write(path, "question_2/training_records/" + path.relative_to(runs).as_posix())
    print(f"Created {output} ({output.stat().st_size / 2**20:.1f} MiB)", flush=True)
    print("This is the Q2 package. Integrate teammates' work and the edited combined report for the group submission.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist" / "q2_source.zip"))
    parser.add_argument("--results")
    parser.add_argument("--config", help="Embed the exact executed configuration instead of the source default")
    parser.add_argument("--runs")
    args = parser.parse_args()
    package(args.output, args.results, args.runs, args.config)
