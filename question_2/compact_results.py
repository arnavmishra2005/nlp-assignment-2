"""Losslessly compact an inspected existing phase ZIP; never overwrites the input."""
import argparse
from pathlib import Path
import shutil
import tempfile
from q2.artifacts import ArtifactReader, inspect_artifact, package_phase
from q2.common import write_json


def compact_existing(archive, output):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if archive == output:
        raise ValueError("Use a different output path; keep the original archive")
    inspect_artifact(archive)
    reader = ArtifactReader(archive)
    try:
        header = reader.json("phase_manifest.json")
        config = reader.json("config.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="q2-compact-") as temporary:
            root = Path(temporary).resolve()
            for name in sorted(reader.names):
                if name == "phase_manifest.json":
                    continue
                target = (root / name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Unsafe extraction path")
                target.parent.mkdir(parents=True, exist_ok=True)
                with reader.open(name) as source, open(target, "wb") as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
            package_phase(root, output, header["phase"], config, header["prepared_manifest_digest"],
                          header["split_fingerprint"], header["source_hash"], header["experiments"])
    finally:
        reader.close()
    print(f"COMPACT_COPY_VERIFIED: {archive.stat().st_size/2**20:.1f} -> {output.stat().st_size/2**20:.1f} MiB", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compact_existing(args.archive, args.output)
