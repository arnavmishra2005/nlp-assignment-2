"""Distributed batches, gradient math, real two-process training and recovery."""
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from q2.common import digest, read_json, write_json, source_hash
from q2.compatibility import compatible_prerequisite
from q2.distributed import RankBatchSampler, topology
from q2.train import EpochBatchSampler

ROOT = Path(__file__).resolve().parents[1]


class DistributedContracts(unittest.TestCase):
    def test_exact_original_batches_and_no_overlap(self):
        for size in (20,22,200000):
            for epoch in (0,2):
                original=list(EpochBatchSampler(size,16,42,epoch,start_batch=1))
                left=list(RankBatchSampler(size,16,42,epoch,0,start_batch=1))
                right=list(RankBatchSampler(size,16,42,epoch,1,start_batch=1))
                self.assertEqual(original,[a+b for a,b in zip(left,right)])
                self.assertTrue(all(not set(a).intersection(b) for a,b in zip(left,right)))
        with self.assertRaisesRegex(ValueError,'no duplication'):
            RankBatchSampler(21,16,42,0,0)

    def test_frozen_hyperparameters_and_update_counts(self):
        config=read_json(ROOT/'config.json')
        plan=topology(config)
        self.assertEqual((plan['world_size'],plan['per_gpu_microbatch'],plan['accumulation_steps'],plan['effective_batch_size']), (2,8,2,32))
        batches=math.ceil(config['dataset']['expected_sizes']['train']/plan['global_microbatch'])
        self.assertEqual(batches,12500)
        self.assertEqual(math.ceil(batches/plan['accumulation_steps']),6250)

    def test_known_completed_runs_remain_compatible_only_for_accepted_phases(self):
        old='5c284bdea9b164be9eb13cfb897f1f8e56cb8ca19bb3d0342ca80362a9d42144'
        for phase in ('baseline_seed42','baseline_seed123'):
            self.assertTrue(compatible_prerequisite({'phase':phase,'source_hash':old},source_hash()))
        self.assertTrue(compatible_prerequisite({'phase':'roberta_mean_seed42','source_hash':old},source_hash()))
        for phase in ('roberta_mean_seed123','roberta_first_seed42','roberta_first_seed123'):
            self.assertFalse(compatible_prerequisite({'phase':phase,'source_hash':old},source_hash()))

    def launch(self, scenario, root, prepared=None, output=None, config=None):
        command=[sys.executable,'-u',str(ROOT/'tests/distributed_worker.py'),'--scenario',scenario,
                 '--rendezvous',str(root/f'{scenario}_{len(list(root.glob("*.log")))}.store'),
                 '--output',str(output or root)]
        if prepared: command+=['--prepared',str(prepared),'--config',str(config)]
        result=subprocess.run(command,capture_output=True,text=True,cwd=ROOT,timeout=240)
        (root/f'{scenario}_{len(list(root.glob("*.log")))}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        self.assertEqual(result.returncode,0,msg=(result.stdout+result.stderr)[-10000:])

    def test_two_process_gradients_match_single_process_with_partial_group(self):
        with tempfile.TemporaryDirectory(prefix='q2-ddp-grad-') as temporary:
            root=Path(temporary)
            self.launch('gradients',root)
            result=torch.load(root/'gradients.pt',weights_only=False)
            self.assertEqual(result['status'],'PASS')

    def test_two_process_roberta_recovery_validation_and_exports(self):
        from test_correctness import fixture_preparation
        from q2.models import load_trained
        from q2.validation import verify_completed_run
        with tempfile.TemporaryDirectory(prefix='q2-ddp-engine-') as temporary:
            root=Path(temporary)
            config,_=fixture_preparation(root/'prepared')
            # Fixture preparation uses an odd batch for single-GPU tail tests.
            # DDP fixture uses a divisible global batch; prepared tokens unchanged.
            config['roberta']['batch_size']=4
            config_path=root/'config.json';write_json(config_path,config)
            recovered=root/'recovered'
            self.launch('interrupt',root,root/'prepared',recovered,config_path)
            recovery=torch.load(recovered/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(len(recovery['rank_states']),2)
            self.assertEqual(recovery['state']['epoch'],0)
            self.assertGreater(recovery['state']['next_batch'],0)
            self.launch('resume',root,root/'prepared',recovered,config_path)
            reference=root/'reference'
            self.launch('reference',root,root/'prepared',reference,config_path)
            a=torch.load(recovered/'best.pt',map_location='cpu',weights_only=False)
            b=torch.load(reference/'best.pt',map_location='cpu',weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key],b['model'][key],rtol=2e-6,atol=2e-7)
                self.assertFalse(key.startswith('module.'))
            evidence=verify_completed_run(recovered)
            self.assertEqual(evidence['run']['distributed']['world_size'],2)
            self.assertEqual(evidence['run']['planned_optimizer_updates'],6)
            self.assertTrue(all(h['train']['n']==20 and h['validation']['n']==10 for h in evidence['history']))
            model,run=load_trained(recovered,torch.device('cpu'))
            self.assertEqual(run['model'],'roberta_mean')
            # Exercise the NCCL-safe spawn DataLoader branch on CPU too.
            worker_config=read_json(config_path)
            worker_config['training']['num_workers']=1
            worker_config_path=root/'worker_config.json'
            write_json(worker_config_path,worker_config)
            self.launch('first',root,root/'prepared',root/'first',worker_config_path)
            self.assertEqual(verify_completed_run(root/'first')['run']['model'],'roberta_first')
            # Actual distributed checkpoint/evidence must pass the existing phase
            # packager and inspector, and export ordinary unwrapped best weights.
            from q2.artifacts import package_phase, inspect_artifact, ArtifactReader
            from q2.common import environment
            payload=root/'payload'
            shutil.copytree(recovered,payload/'runs/roberta_mean_seed42')
            write_json(payload/'config.json',config)
            write_json(payload/'environment/environment.json',environment())
            (payload/'environment/environment-freeze.txt').write_text('CPU GLOO FIXTURE',encoding='utf-8')
            manifest=read_json(root/'prepared/manifest.json')
            fingerprint=read_json(root/'prepared/audit/audit.json')['split_fingerprint']
            archive=root/'Q2_roberta_mean_seed42_results.zip'
            package_phase(payload,archive,'roberta_mean_seed42',config,digest(manifest),fingerprint,source_hash(),['roberta_mean_seed42'])
            self.assertEqual(inspect_artifact(archive,expected_source=source_hash())['status'],'PASS')
            reader=ArtifactReader(archive)
            self.assertIn('runs/roberta_mean_seed42/distributed_execution.json',reader.names)
            self.assertNotIn('runs/roberta_mean_seed42/last.pt',reader.names)
            reader.close()


if __name__=='__main__':
    unittest.main()
