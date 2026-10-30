"""Estimate remaining capped-epoch work from a returned phase archive's timing evidence."""
import argparse
import json
from q2.artifacts import ArtifactReader
from q2.timing import estimate_remaining

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--overhead-minutes", type=float, default=15)
    args = parser.parse_args()
    reader = ArtifactReader(args.archive)
    try:
        evidence = reader.json("timing_evidence.json")
        print(json.dumps(estimate_remaining(reader.json("config.json"), evidence["records"], args.overhead_minutes,
            target_world_sizes={'baseline':1,'roberta_mean':2,'roberta_first':2}), indent=2))
    finally:
        reader.close()
