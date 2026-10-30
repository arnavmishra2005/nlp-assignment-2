"""Storage changes must preserve model tensors and unfinished recovery state."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from q2.common import atomic_torch_save, file_hash, source_hash
from q2.compatibility import compatible_prerequisite, LEGACY
from q2.storage import completed_storage_policy


class StorageContracts(unittest.TestCase):
    def test_smoke_optimizer_omission_preserves_exact_model(self):
        with tempfile.TemporaryDirectory(prefix="q2-storage-") as temporary:
            root = Path(temporary)
            path = root / "smoke/roberta_mean/smoke_checkpoint.pt"
            state = {"weight": torch.randn(4, 7), "counter": torch.tensor(5, dtype=torch.int64)}
            original = {"model": state, "optimizer": {"moment": torch.ones(100)}, "scheduler": {"step": 2},
                        "scaler": {}, "rng": {}, "metadata": {"encoder": "fixture"}, "config": {"test": True}}
            atomic_torch_save(original, path)
            completed_storage_policy(root, "smoke", "COMPLETE")
            actual = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("optimizer", actual)
            for name, tensor in state.items():
                self.assertEqual(tensor.dtype, actual["model"][name].dtype)
                self.assertTrue(torch.equal(tensor, actual["model"][name]))

    def test_failed_phase_keeps_full_recovery_bytes(self):
        with tempfile.TemporaryDirectory(prefix="q2-recovery-storage-") as temporary:
            root = Path(temporary)
            path = root / "runs/baseline_seed42/last.pt"
            atomic_torch_save({"optimizer": {"step": 12}, "rng": torch.get_rng_state()}, path)
            before = file_hash(path)
            policy = completed_storage_policy(root, "baseline_seed42", "FAILED")
            self.assertEqual(policy["excluded_files"], [])
            self.assertEqual(file_hash(path), before)
            self.assertIn("optimizer", torch.load(path, weights_only=False))

    def test_legacy_allowance_requires_exact_core_and_phase(self):
        header = {"phase": "prepare", "source_hash": LEGACY["legacy_source_hash"]}
        self.assertTrue(compatible_prerequisite(header, source_hash()))
        header["phase"] = "baseline_seed42"
        self.assertTrue(compatible_prerequisite(header, source_hash()))
        header["phase"] = "baseline_seed123"
        self.assertFalse(compatible_prerequisite(header, source_hash()))
        header["phase"] = "smoke"
        with patch("q2.compatibility.file_hash", return_value="changed core"):
            self.assertFalse(compatible_prerequisite(header, source_hash()))


if __name__ == "__main__":
    unittest.main()
