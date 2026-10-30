"""Meaningful contract checks for new artifact inspection and staged notebooks."""
import ast
import json
from pathlib import Path
import tempfile
import contextlib
import io
import subprocess
import sys
import unittest
import zipfile
import numpy as np
import pandas as pd
from q2.artifacts import ArtifactReader, check_predictions
from q2.common import read_json
from q2.metrics import classification_metrics
from q2.timing import estimate_remaining

ROOT = Path(__file__).resolve().parents[1]


class WorkflowContracts(unittest.TestCase):
    def test_single_gpu_timings_do_not_predict_dual_gpu_duration(self):
        from q2.common import load_config
        config=load_config(ROOT/'config.json')
        records={name:{'complete':True,'epoch_seconds':[100]} for name in ('baseline_seed42','baseline_seed123')}
        records['roberta_mean_seed42']={'complete':True,'epoch_seconds':[7700],'world_size':1}
        targets={'baseline':1,'roberta_mean':2,'roberta_first':2}
        estimate=estimate_remaining(config,records,target_world_sizes=targets)
        self.assertFalse(estimate['whole_pipeline_estimate_available'])
        self.assertIn('roberta_mean_seed123',estimate['unknown_model_timings'])
        records['roberta_mean_seed123']={'complete':True,'epoch_seconds':[4000],'world_size':2}
        first=estimate_remaining(config,records,target_world_sizes=targets)
        self.assertTrue(first['whole_pipeline_estimate_available'])
        records['roberta_mean_seed42']['epoch_seconds']=[999999]
        second=estimate_remaining(config,records,target_world_sizes=targets)
        self.assertEqual(first['known_work_range_hours_at_epoch_caps'],second['known_work_range_hours_at_epoch_caps'])

    def test_actual_topology_and_mixed_execution_disclosure(self):
        from q2.execution import execution_details, execution_note
        from q2.common import load_config
        run={'model':'roberta_mean','source_hash':'original-source','config':load_config(ROOT/'config.json'),
             'environment':{'gpu_count':2}}
        first=execution_details(run)
        self.assertEqual(first['world_size'],1)  # Two allocated GPUs did not train this original run.
        self.assertEqual(first['effective_batch_size'],32)
        run['distributed']={'strategy':'DistributedDataParallel','world_size':2,'per_gpu_microbatch':8}
        second=execution_details(run)
        self.assertEqual(second['per_device_microbatch'],8)
        self.assertEqual(second['effective_batch_size'],32)
        note=execution_note([{'model':'roberta_mean',**first},{'model':'roberta_first',**second}])
        self.assertIn('not fully controlled for execution topology',note)

    def test_plain_text_progress_in_fresh_process(self):
        code = "import q2; from tqdm.auto import tqdm; bar=tqdm(range(3),desc='UNWANTED_BAR'); assert bar.disable; list(bar); print('PLAIN_TEXT_ONLY')"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'PLAIN_TEXT_ONLY')
        self.assertEqual(result.stderr, '')

    def test_old_source_rejected_before_phase_launch(self):
        from tools.make_notebooks import SETUP
        guard = SETUP[SETUP.index('required_files ='):SETUP.index('if str(PROJECT) not in sys.path:')]
        with tempfile.TemporaryDirectory(prefix='q2-source-gate-') as temporary:
            project = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, 'OUTDATED_SOURCE_DATASET'):
                exec(guard, {'PROJECT': project})
            for name in ('deliver_phase.py', 'compact_results.py', 'q2/storage.py', 'q2/compatibility.py',
                         'q2/distributed.py', 'q2/train_distributed.py'):
                path = project/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fixture', encoding='utf-8')
            (project/'q2/__init__.py').write_text('# old package', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'plain-text logging'):
                exec(guard, {'PROJECT': project})
            (project/'q2/__init__.py').write_text("# TQDM_DISABLE", encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                exec(guard, {'PROJECT': project})

    def test_probability_and_identifier_checks(self):
        frame = pd.DataFrame({"id": [f"sample{i}" for i in range(5)], "true_stars": range(1, 6), "predicted_stars": range(1, 6)})
        for i in range(5):
            frame[f"probability_{i+1}"] = np.eye(5)[:, i]
        metrics = classification_metrics(np.arange(5), np.arange(5))
        class Reader:
            def csv(self, path):
                return frame
        check_predictions(Reader(), "fixture.csv", metrics, 5)
        frame.loc[0, "probability_1"] = 2.0
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            check_predictions(Reader(), "fixture.csv", metrics, 5)
        frame.loc[0, "probability_1"] = 1.0
        frame.loc[1, "id"] = "sample0"
        with self.assertRaisesRegex(ValueError, "duplicate/missing IDs"):
            check_predictions(Reader(), "fixture.csv", metrics, 5)

    def test_unsafe_archive_rejected(self):
        with tempfile.TemporaryDirectory(prefix="q2-archive-contract-") as temporary:
            path = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../outside.txt", "untrusted")
                archive.writestr("Q2_prepare/phase_manifest.json", "{}")
            with self.assertRaisesRegex(ValueError, "Unsafe archive entry"):
                ArtifactReader(path)

    def test_model_specific_time_estimates(self):
        config = read_json(ROOT / "config.json")
        records = {"baseline_seed42": {"complete": True, "epoch_seconds": [10, 12, 11]}}
        estimate = estimate_remaining(config, records)
        self.assertFalse(estimate["whole_pipeline_estimate_available"])
        self.assertNotIn("approximate_finish_range_ist_at_caps", estimate)
        records["roberta_mean_seed42"] = {"complete": True, "epoch_seconds": [30, 32]}
        estimate = estimate_remaining(config, records)
        self.assertTrue(estimate["whole_pipeline_estimate_available"])
        self.assertTrue(any("proxy" in note for note in estimate["assumptions"]))
        self.assertIn("+05:30", estimate["approximate_finish_range_ist_at_caps"][0])

    def test_streaming_heartbeat_and_failed_exit(self):
        from q2.process import run_streaming
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_streaming([sys.executable, "-u", "-c", "import time; time.sleep(1.15); print('done')"], heartbeat_seconds=.1)
        self.assertIn("heartbeat", output.getvalue())
        self.assertIn("does not prove useful progress", output.getvalue())
        self.assertIn("done", output.getvalue())
        with self.assertRaises(subprocess.CalledProcessError):
            run_streaming([sys.executable, "-c", "raise SystemExit(3)"])

    def test_notebooks_have_three_valid_code_cells(self):
        notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 11)
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            self.assertEqual(len(cells), 3)
            for cell in cells:
                ast.parse("".join(cell["source"]), filename=path.name)
                self.assertEqual(cell["outputs"], [])
            tree = ast.parse("".join(cells[0]["source"]))
            node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "probe" for t in n.targets))
            probe = eval(compile(ast.Expression(node.value), "probe", "eval"))
            ast.parse(probe)


if __name__ == "__main__":
    unittest.main()
