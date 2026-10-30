"""Publish exactly one inspected phase ZIP to Kaggle's saved output directory.

Executable source and temporary artifacts belong outside /kaggle/working. This
script copies only the verified archive; it never deletes unrelated user files.
"""
import argparse
from pathlib import Path
import shutil
from q2.artifacts import ArtifactReader, inspect_artifact
from q2.common import file_hash


def deliver(archive, destination):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    reader = ArtifactReader(archive)
    try:
        metadata = reader.json("phase_manifest.json")
    finally:
        reader.close()
    failed = metadata["status"] == "FAILED"
    inspect_artifact(archive, expected_phase=metadata["phase"], allow_incomplete=failed)
    destination.mkdir(parents=True, exist_ok=True)
    filename = f"Q2_{metadata['phase']}_{'FAILED' if failed else 'results'}.zip"
    target = destination / filename
    if target.exists():
        if file_hash(target) != file_hash(archive):
            raise ValueError("A different result ZIP already exists; start a fresh notebook session")
        return target
    temporary = target.with_suffix(".zip.partial")
    shutil.copyfile(archive, temporary)
    if file_hash(temporary) != file_hash(archive):
        raise ValueError("Published ZIP differs from the inspected archive")
    inspect_artifact(temporary, expected_phase=metadata["phase"], allow_incomplete=failed)
    temporary.replace(target)
    print("ONE_PROJECT_OUTPUT_ZIP:", target, flush=True)
    print("Download this individual ZIP. Kaggle's Download All may wrap outputs in another archive.", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", default="/kaggle/working")
    args = parser.parse_args()
    deliver(args.archive, args.destination)
