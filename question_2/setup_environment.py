"""Check compatibility first; install only missing/incompatible non-CUDA dependencies."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def ensure_environment(install=False, inference=False):
    from packaging.requirements import Requirement
    from packaging.version import Version
    print("Python:", sys.version, flush=True)
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >=3.10 is required")
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("PyTorch is missing. Use Kaggle's GPU environment; this script will not install a CUDA stack.")
    if not Version("2.5") <= Version(torch_version.split("+")[0]) < Version("3"):
        raise RuntimeError(f"Unsupported PyTorch {torch_version}; need >=2.5,<3; do not blindly replace CUDA")
    print("Preserving installed torch:", torch_version, flush=True)
    path = Path(__file__).parent / ("requirements_inference.txt" if inference else "requirements_training.txt")
    missing = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        compatible = installed is not None and requirement.specifier.contains(installed)
        print(f"dependency {requirement.name}: {installed or 'MISSING'} | required {requirement.specifier} | {'OK' if compatible else 'CHANGE NEEDED'}", flush=True)
        if not compatible:
            missing.append(line)
    if missing:
        if not install:
            raise RuntimeError("Dependencies need installation: " + ", ".join(missing) + ". Run setup_environment.py --install-missing.")
        # The request excludes torch entirely. Pin the already-installed CUDA stack
        # via constraints so transitive dependency resolution cannot replace it.
        constraint = Path(__file__).parent / "environment-constraints.txt"
        constraint.write_text("torch==" + torch_version + "\n", encoding="utf-8")
        subprocess.run([sys.executable, "-u", "-m", "pip", "install", "--disable-pip-version-check",
                        "--constraint", str(constraint), *missing], check=True)
    # This process is a fresh subprocess, so upgrades are imported by the next one.
    actual = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    print("DEPENDENCIES_READY", json.dumps(actual), flush=True)
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-missing", action="store_true")
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()
    ensure_environment(args.install_missing, args.inference)
