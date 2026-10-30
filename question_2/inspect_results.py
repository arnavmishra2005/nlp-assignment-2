"""Inspect a returned phase ZIP/directory before proceeding to the next phase."""
import argparse
from q2.artifacts import inspect_artifact
from q2.common import load_config, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--expected-phase")
    parser.add_argument("--allow-incomplete", action="store_true", help="Integrity-check failure artifacts for recovery only")
    parser.add_argument("--config", help="Optional expected config; omit to inspect the embedded configuration")
    parser.add_argument("--receipt", help="Write a machine-readable PASS receipt")
    args = parser.parse_args()
    result = inspect_artifact(args.archive, args.expected_phase, load_config(args.config) if args.config else None, allow_incomplete=args.allow_incomplete)
    if args.receipt:
        write_json(args.receipt, result)
