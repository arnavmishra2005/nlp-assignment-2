"""CPU synthetic correctness tests. No assignment data or pretrained downloads."""
import copy
from contextlib import contextmanager
import logging
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from q2.common import digest, file_hash, read_json, write_json
from q2.data import build_embeddings, classic_tokens, encode_classic, join_title_body, preparation_key
from q2.metrics import classification_metrics
from q2.models import BiGRUAttention, masked_mean
from q2.train import EpochBatchSampler, run_training

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def close_logs_on_exit():
    try:
        yield
    finally:
        log = logging.getLogger("q2")
        for handler in list(log.handlers):
            log.removeHandler(handler)
            handler.close()


class CoreCorrectness(unittest.TestCase):
    def test_classic_cleaning_and_title_validation(self):
        self.assertEqual(classic_tokens("<b>NOT</b> bad! https://example.com"), ["not", "bad", "!"])
        ids, lengths, _ = encode_classic(["new unseen", ""], {"<pad>": 0, "<unk>": 1}, 3)
        self.assertEqual(ids.tolist(), [[1, 1, 0], [1, 0, 0]])
        self.assertEqual(lengths.tolist(), [2, 1])
        raw = [{"id": f"en_{i}", "text": f"body {i}", "label": i} for i in range(5)]
        source = [{"review_id": f"en_{i}", "review_body": f"body {i}", "review_title": "Title",
                   "stars": i + 1, "language": "en"} for i in range(5)]
        joined = join_title_body(raw, list(reversed(source)), 5)
        self.assertEqual(joined.text.iloc[0], "Title\nbody 0")
        source[0]["review_title"] = None
        self.assertEqual(join_title_body(raw, source, 5).text.iloc[0], "body 0")
        source[0]["stars"] = 5
        with self.assertRaisesRegex(ValueError, "Label mismatch"):
            join_title_body(raw, source, 5)

    def test_pretrained_embedding_loading(self):
        with tempfile.TemporaryDirectory(prefix="q2-vectors-") as temporary:
            root = Path(temporary)
            vectors = root / "vectors.txt"
            vectors.write_text("good 0.1 0.2 0.3\nbad -0.1 -0.2 -0.3\nother 1 2 3\n", encoding="utf-8")
            stats = build_embeddings({"<pad>": 0, "<unk>": 1, "good": 2, "bad": 3}, 3,
                                     root, vectors, logging.getLogger("q2-embedding-test"))
            matrix = np.load(root / "embeddings.npy")
            np.testing.assert_allclose(matrix[2], [.1, .2, .3])
            np.testing.assert_array_equal(matrix[0], np.zeros(3))
            self.assertEqual(stats["vocabulary_coverage"], 1.0)

    def test_masked_pooling_and_packed_gru(self):
        states = torch.tensor([[[1., 3.], [3., 5.], [900., 900.]]])
        pooled = masked_mean(states, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(pooled, torch.tensor([[2., 4.]]))
        config = {"hidden_dim": 4, "attention_dim": 3, "num_layers": 2, "dropout": 0.0}
        model = BiGRUAttention(torch.randn(8, 5), config).eval()
        short = torch.tensor([[2, 3, 4]])
        long = torch.tensor([[2, 3, 4, 0, 0, 0]])
        first, _ = model(short, torch.tensor([3]), return_attention=True)
        second, attention = model(long, torch.tensor([3]), return_attention=True)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(attention[:, 3:], torch.zeros(1, 3))
        torch.testing.assert_close(attention.sum(1), torch.ones(1))

    def test_metrics_respect_ordinal_distance(self):
        result = classification_metrics([0, 1, 2, 3, 4], [0, 2, 0, 4, 0])
        self.assertAlmostEqual(result["accuracy"], .2)
        self.assertAlmostEqual(result["mae"], 1.6)
        self.assertAlmostEqual(result["adjacent_error_rate"], .4)
        self.assertAlmostEqual(result["distant_error_rate"], .4)
        self.assertAlmostEqual(result["adjacent_share_of_errors"], .5)
        self.assertAlmostEqual(result["extreme_1_vs_5_error_rate"], .2)
        perfect = classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertEqual(perfect["macro_f1"], 1)
        self.assertEqual(perfect["mae"], 0)

    def test_shuffle_recovery_and_partial_accumulation(self):
        original = list(EpochBatchSampler(23, 4, 42, 2))
        resumed = list(EpochBatchSampler(23, 4, 42, 2, start_batch=3))
        self.assertEqual(original[3:], resumed)
        self.assertEqual(sorted(x for batch in original for x in batch), list(range(23)))
        # Sum-reduction / actual group size must match a single combined batch gradient.
        torch.manual_seed(9)
        reference = torch.nn.Linear(3, 5)
        accumulated = copy.deepcopy(reference)
        x, y = torch.randn(7, 3), torch.tensor([0, 1, 2, 3, 4, 0, 1])
        torch.nn.functional.cross_entropy(reference(x), y).backward()
        for begin, end in ((0, 4), (4, 7)):
            (torch.nn.functional.cross_entropy(accumulated(x[begin:end]), y[begin:end], reduction="sum") / 7).backward()
        for left, right in zip(reference.parameters(), accumulated.parameters()):
            torch.testing.assert_close(left.grad, right.grad)


def fixture_preparation(path):
    from datasets import Dataset
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
    path.mkdir(parents=True)
    config = read_json(ROOT / "config.json")
    config["dataset"]["expected_sizes"] = {"train": 20, "validation": 10, "test": 10}
    config["baseline"].update({"max_length": 12, "hidden_dim": 4, "attention_dim": 4,
         "embedding_dim": 8, "batch_size": 4, "eval_batch_size": 5, "epochs": 2, "patience": 2})
    config["roberta"].update({"max_length": 16, "batch_size": 3, "eval_batch_size": 5,
                              "epochs": 2, "patience": 2, "precision": "fp32"})
    config["training"].update({"num_workers": 0, "checkpoint_every_optimizer_steps": 1,
                               "log_every_batches": 100, "min_delta": 0.0})
    frames = {}
    for split, size in config["dataset"]["expected_sizes"].items():
        frames[split] = pd.DataFrame([{"id": f"{split}_{i:03}", "title": "Review",
            "body": ["very bad", "poor", "average", "good", "excellent"][i % 5],
            "text": "Review\n" + ["very bad", "poor", "average", "good", "excellent"][i % 5],
            "label": i % 5} for i in range(size)])
    corpus = frames["train"].text.tolist()
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in corpus:
        for token in classic_tokens(text):
            vocab.setdefault(token, len(vocab))
    write_json(path / "vocab.json", vocab)
    embeddings = np.random.default_rng(1).normal(0, .05, (len(vocab), 8)).astype("float32")
    embeddings[0] = 0
    np.save(path / "embeddings.npy", embeddings)
    token_dir = path / "tokenizer"
    token_dir.mkdir()
    trainer = ByteLevelBPETokenizer()
    trainer.train_from_iterator(corpus, vocab_size=280, min_frequency=1,
                               special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    trainer.save_model(str(token_dir))
    tokenizer = RobertaTokenizerFast(vocab_file=str(token_dir / "vocab.json"), merges_file=str(token_dir / "merges.txt"))
    tokenizer.save_pretrained(token_dir)
    model = RobertaModel(RobertaConfig(vocab_size=len(tokenizer), hidden_size=12, num_hidden_layers=1,
            num_attention_heads=3, intermediate_size=24, max_position_embeddings=34, pad_token_id=1), add_pooling_layer=False)
    model.save_pretrained(path / "pretrained")
    tokenizer.save_pretrained(path / "pretrained")
    for split in ("train", "validation"):
        frame = frames[split]
        frame.to_parquet(path / f"{split}.parquet", index=False)
        ids, lengths, _ = encode_classic(frame.text.tolist(), vocab, 12)
        np.savez_compressed(path / f"baseline_{split}.npz", input_ids=ids, lengths=lengths, labels=frame.label.to_numpy())
        encoded = tokenizer(frame.text.tolist(), truncation=True, max_length=16)
        Dataset.from_dict({**encoded, "labels": frame.label.tolist()}).save_to_disk(str(path / f"roberta_{split}"))
    write_json(path / "data_statistics.json", {"exact_train_validation_text_overlap": 0,
               "vocabulary_size": len(vocab), "embeddings": {"vocabulary_coverage": .8}})
    from q2.audit import audit_frames
    audit_frames({key: frames[key] for key in ("train", "validation")}, config, path / "audit")
    checksums = {p.relative_to(path).as_posix(): file_hash(p) for p in path.rglob("*") if p.is_file()}
    write_json(path / "manifest.json", {"preparation_key": preparation_key(config), "files": checksums, "test_accessed": False})
    return config, frames


class EndToEndCorrectness(unittest.TestCase):
    def test_training_recovery_and_complete_analysis(self):
        from q2.evaluate import evaluate_final
        from q2.common import atomic_torch_save
        with tempfile.TemporaryDirectory(prefix="q2-correctness-") as temporary, close_logs_on_exit():
            root = Path(temporary)
            prepared = root / "prepared"
            config, frames = fixture_preparation(prepared)
            runs = root / "runs"
            # Interrupt at a saved optimizer boundary midway through the first epoch.
            interrupted = runs / "baseline_seed42"
            def interrupt_at_checkpoint(value, path):
                atomic_torch_save(value, path)
                if Path(path).name == "last.pt" and value["state"]["epoch"] == 0 and value["state"]["next_batch"] == 2:
                    raise RuntimeError("synthetic interruption")
            with patch("q2.train.atomic_torch_save", side_effect=interrupt_at_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    run_training(config, prepared, interrupted, "baseline", 42, allow_cpu=True)
            self.assertTrue((interrupted / "last.pt").exists())
            run_training(config, prepared, interrupted, "baseline", 42, allow_cpu=True)
            reference = root / "reference"
            run_training(config, prepared, reference, "baseline", 42, allow_cpu=True)
            left = torch.load(interrupted / "best.pt", map_location="cpu", weights_only=False)
            right = torch.load(reference / "best.pt", map_location="cpu", weights_only=False)
            for name in left["model"]:
                torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
            # Frozen first-epoch weights become trainable and update in epoch two.
            last = torch.load(interrupted / "last.pt", map_location="cpu", weights_only=False)
            initial = torch.from_numpy(np.load(prepared / "embeddings.npy"))
            self.assertFalse(torch.equal(last["model"]["embedding.weight"][1:], initial[1:]))
            for model in ("baseline", "roberta_mean", "roberta_first"):
                for seed in config["seeds"]:
                    if model == 'roberta_mean' and seed == 42:
                        # Actual unchanged single-GPU trainer with the historical
                        # source identity: verify retained-run evaluation/export.
                        with patch('q2.train.source_hash', return_value='5c284bdea9b164be9eb13cfb897f1f8e56cb8ca19bb3d0342ca80362a9d42144'):
                            run_training(config, prepared, runs / f"{model}_seed{seed}", model, seed, allow_cpu=True)
                    else:
                        run_training(config, prepared, runs / f"{model}_seed{seed}", model, seed, allow_cpu=True)
            # The only test loader is patched to an in-memory synthetic split.
            with patch("q2.evaluate.load_split", return_value=frames["test"]) as loader:
                evaluate_final(config, prepared, runs, root / "final", allow_cpu=True)
                loader.assert_called_once()
            final = root / "final"
            self.assertTrue((final / "EVALUATION_COMPLETE.json").exists())
            self.assertTrue((final / "figures/roberta_token_importance.png").exists())
            self.assertNotIn("nan", (final / "test_summary.csv").read_text().lower())
            self.assertIn("TODO", (final / "REPORT_DRAFT.md").read_text(encoding="utf-8"))
            protocol=read_json(final/'FROZEN_PROTOCOL.json')
            retained=next(r for r in protocol['runs'] if r['model']=='roberta_mean' and r['seed']==42)
            self.assertEqual(retained['training_execution']['world_size'],1)
            self.assertEqual(retained['training_execution']['source_hash'],'5c284bdea9b164be9eb13cfb897f1f8e56cb8ca19bb3d0342ca80362a9d42144')
            with patch("q2.evaluate.load_split", side_effect=AssertionError("test reread")), \
                 patch("q2.evaluate.evaluate_loader", side_effect=AssertionError("test rescored")):
                evaluate_final(config, prepared, runs, final, allow_cpu=True)
            # Exercise the actual archive contract, smoke phase, setup gates and finalizer.
            from q2.artifacts import inspect_artifact, package_phase, TRAIN_PHASES
            from q2.common import environment, source_hash
            from q2.smoke import run_smoke
            from q2.workflow import launch
            header = read_json(prepared / "manifest.json")
            fingerprint = read_json(prepared / "audit/audit.json")["split_fingerprint"]
            def payload_for(phase):
                payload = root / "payloads" / phase
                payload.mkdir(parents=True)
                write_json(payload / "config.json", config)
                write_json(payload / "environment/environment.json", environment())
                (payload / "environment/environment-freeze.txt").write_text("SYNTHETIC CPU FIXTURE", encoding="utf-8")
                return payload
            def pack(payload, phase, experiments, status="COMPLETE"):
                archive = root / "archives" / f"Q2_{phase}_results.zip"
                package_phase(payload, archive, phase, config, digest(header), fingerprint, source_hash(), experiments, status)
                return archive
            prepare_payload = payload_for("prepare")
            shutil.copytree(prepared, prepare_payload / "prepared")
            prepare_zip = pack(prepare_payload, "prepare", [])
            smoke_payload = payload_for("smoke")
            run_smoke(config, prepared, smoke_payload / "smoke", allow_cpu=True)
            smoke_zip = pack(smoke_payload, "smoke", [])
            self.assertEqual(inspect_artifact(smoke_zip)["status"], "PASS")
            from q2.artifacts import ArtifactReader
            reader = ArtifactReader(smoke_zip)
            self.assertIn("smoke/checkpoint_storage.json", reader.names)
            self.assertTrue(all(item["every_model_tensor_equal"] for item in reader.json("smoke/checkpoint_storage.json")["changes"]))
            reader.close()
            training_archives = {}
            for phase in TRAIN_PHASES:
                payload = payload_for(phase)
                shutil.copytree(runs / phase, payload / "runs" / phase)
                training_archives[phase] = pack(payload, phase, [phase])
                reader = ArtifactReader(training_archives[phase])
                self.assertIn(f"runs/{phase}/best.pt", reader.names)
                self.assertNotIn(f"runs/{phase}/last.pt", reader.names)
                reader.close()
            evaluation_payload = payload_for("evaluate")
            shutil.copytree(final, evaluation_payload / "final")
            shutil.copytree(runs, evaluation_payload / "runs", ignore=shutil.ignore_patterns("last.pt"))
            evaluation_zip = pack(evaluation_payload, "evaluate", TRAIN_PHASES)
            reader = ArtifactReader(evaluation_zip)
            self.assertFalse(any(name.endswith(".pt") for name in reader.names))
            self.assertEqual(len(reader.json("phase_manifest.json")["storage_policy"]["checkpoint_references"]), 6)
            reader.close()
            fixture_config = root / "fixture_config.json"
            write_json(fixture_config, config)
            setup_options = SimpleNamespace(phase="baseline_seed42", config=str(fixture_config), output=str(root / "setup-check"),
                setup_only=True, data=str(prepare_zip), previous=[str(smoke_zip)], inputs=str(root / "empty-input"),
                seed=42, resume_from=None, allow_cpu=True, min_free_gib=.01, resume=True, glove_path=None, keep_cache=False)
            planned = launch(setup_options)
            self.assertEqual(planned["current_run"]["seed"], 42)
            self.assertFalse((root / "setup-check/baseline_seed42/payload/runs").exists())
            setup_options.output = str(root / "missing-prerequisite")
            setup_options.previous = []
            with self.assertRaisesRegex(ValueError, "Expected exactly one smoke"):
                launch(setup_options)
            options = SimpleNamespace(phase="finalize", config=str(fixture_config), output=str(root / "workflow"),
                setup_only=False, data=str(prepare_zip), previous=[str(evaluation_zip)], inputs=str(root / "empty-input"),
                seed=None, resume_from=None, allow_cpu=True, min_free_gib=.01, resume=True, glove_path=None, keep_cache=False)
            result = launch(options)
            self.assertEqual(result["status"], "PASS")
            execution = read_json(root / "workflow/finalize/execution.json")
            self.assertEqual(execution["status"], "COMPLETE")
            from deliver_phase import deliver
            published = root / "published"
            delivered = deliver(root / "workflow/finalize/Q2_finalize_results.zip", published)
            self.assertEqual([p.name for p in published.iterdir()], ["Q2_finalize_results.zip"])
            reader = ArtifactReader(delivered)
            self.assertIn("question_2/README.md", reader.names)
            self.assertNotIn("question_2_submission.zip", reader.names)
            reader.close()
            # A second invocation verifies and reuses the completed phase.
            self.assertEqual(launch(options)["status"], "PASS")
            failure_payload = payload_for("recovery")
            failed_zip = pack(failure_payload, "baseline_seed42", ["baseline_seed42"], status="FAILED")
            with self.assertRaisesRegex(ValueError, "has not completed"):
                inspect_artifact(failed_zip)
            self.assertEqual(inspect_artifact(failed_zip, allow_incomplete=True)["status"], "RECOVERY_ONLY")
            # The optional prediction CLI must actually execute, not just parse.
            output_json = root / "cli_prediction.json"
            command = [sys.executable, str(ROOT / "predict.py"), "--run", str(runs / "roberta_mean_seed42"),
                       "--prepared", str(prepared), "--text", "Review\nThe item works well.", "--device", "cpu", "--output", str(output_json)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            prediction = read_json(output_json)
            self.assertIn(prediction["predicted_stars"], range(1, 6))
            self.assertAlmostEqual(sum(prediction["probabilities_1_to_5"]), 1, places=5)
            from predict import predict
            with self.assertRaisesRegex(ValueError, "nonempty"):
                predict(runs / "roberta_mean_seed42", prepared, " ")
            # Replaced checkpoint files must be caught before any new test access.
            with open(runs / "baseline_seed42/best.pt", "ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Corrupt or replaced"):
                evaluate_final(config, prepared, runs, final, allow_cpu=True)


if __name__ == "__main__":
    unittest.main()
