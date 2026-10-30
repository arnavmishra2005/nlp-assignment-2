"""Audit the official train/validation splits; never manufacture a new test split."""
from collections import Counter
import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .common import file_hash, write_json


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def audit_frames(frames, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, reports = [], {}
    for split, frame in frames.items():
        if split == "test":
            raise ValueError("Preparation audit must not read the reserved test split")
        expected = config["dataset"]["expected_sizes"][split]
        if len(frame) != expected or frame.id.duplicated().any():
            raise ValueError(f"Wrong size or duplicate review IDs in {split}")
        if frame.text.isna().any() or frame.text.str.strip().eq("").any():
            raise ValueError(f"Missing/empty model input in {split}")
        counts = frame.label.value_counts().sort_index().to_dict()
        if counts != {label: expected // 5 for label in range(5)}:
            raise ValueError(f"Unexpected official balanced label counts in {split}: {counts}")
        reports[split] = {"rows": len(frame), "class_counts": counts,
            "missing_titles": int(frame.title.fillna("").eq("").sum()),
            "missing_bodies": int(frame.body.fillna("").eq("").sum()),
            "exact_duplicate_extra_rows": int(frame.text.duplicated().sum()),
            "rating_mentions": int(frame.text.str.contains(r"\b[1-5][ -]?stars?\b", case=False, regex=True).sum()),
            "schema": {column: str(dtype) for column, dtype in frame.dtypes.items()}}
        for record in frame.to_dict("records"):
            rows.append({"id": record["id"], "split": split, "label": int(record["label"]),
                         "text_sha256": text_sha(record["text"]), "normalized_sha256": text_sha(canonical(record["text"])),
                         "product_id": record.get("product_id"), "reviewer_id": record.get("reviewer_id")})
    inventory = pd.DataFrame(rows)
    if inventory.id.duplicated().any():
        raise ValueError("Review IDs overlap across the official training/validation splits")
    exact_groups = inventory.groupby("text_sha256")
    conflicts = inventory.groupby("text_sha256").label.nunique()
    cross_exact = exact_groups.split.nunique()
    overlap = {"review_id_overlap": 0,
               "exact_text_groups_across_splits": int(cross_exact.gt(1).sum()),
               "exact_text_groups_with_inconsistent_labels": int(conflicts.gt(1).sum())}
    for group in ("product_id", "reviewer_id"):
        if inventory[group].notna().any():
            counts = inventory.dropna(subset=[group]).groupby(group).size()
            overlap[group] = {"available": True, "shared_across_splits": int(inventory.dropna(subset=[group]).groupby(group).split.nunique().gt(1).sum()),
                              "largest_group": int(counts.max()), "largest_10_group_sizes": counts.nlargest(10).tolist()}
        else:
            overlap[group] = {"available": False}
    # Normalization proposes candidates only. Verify the actual strings rather
    # than chaining similarity matches or changing the assignment's fixed splits.
    texts = {record["id"]: record["text"] for frame in frames.values() for record in frame.to_dict("records")}
    candidates = []
    for _, group in inventory.groupby("normalized_sha256", sort=False):
        if len(group) < 2:
            continue
        first = group.iloc[0]
        for _, second in group.iloc[1:11].iterrows():
            if first.text_sha256 == second.text_sha256:
                continue
            ratio = SequenceMatcher(None, texts[first.id], texts[second.id], autojunk=False).ratio()
            candidates.append({"id_a": first.id, "id_b": second.id, "split_a": first.split, "split_b": second.split,
                "character_similarity": ratio, "verified_near_duplicate": ratio >= .95})
            if len(candidates) >= 200:
                break
        if len(candidates) >= 200:
            break
    pd.DataFrame(candidates, columns=["id_a", "id_b", "split_a", "split_b", "character_similarity", "verified_near_duplicate"]).to_csv(
        output / "near_duplicate_candidates.csv", index=False)
    inventory.to_csv(output / "split_manifest.csv", index=False)
    duplicate_ids = inventory[inventory.text_sha256.isin(set(conflicts[conflicts.gt(1)].index))]
    duplicate_ids.to_csv(output / "label_conflict_candidates.csv", index=False)
    result = {"status": "PASS", "policy": "Official assignment splits preserved; no automatic removal, regrouping or resplitting",
        "class_mapping": {str(i): f"{i+1} stars" for i in range(5)}, "dataset": config["dataset"], "splits": reports,
        "overlap": overlap, "near_duplicate_search": {"method": "case/punctuation/whitespace normalization proposes candidates; verify character similarity >=0.95",
            "candidates_checked": len(candidates), "cap": 200, "exhaustive_semantic_search": False,
            "note": "Similarity is diagnostic; no transitive merging or data deletion"},
        "split_fingerprint": file_hash(output / "split_manifest.csv"),
        "limitations": ["Official corpus overlaps are disclosed rather than silently changing required splits",
                         "Product/reviewer overlap is audited, not assumed to be identity-independent deployment",
                         "Rating mentions may be legitimate review text; no automatic shortcut claim",
                         "Reserved test schema, IDs and counts are checked only after protocol freezing"]}
    write_json(output / "audit.json", result)
    write_json(output / "split_policy.json", {"policy": result["policy"], "split_fingerprint": result["split_fingerprint"],
        "training_only_preprocessing": True, "validation_for_selection": True, "test_reserved": True})
    print("AUDIT_PASSED", reports, overlap, flush=True)
    return result
