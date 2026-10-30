"""Spawned CPU/Gloo integration checks; no dataset download or GPU allocation."""
import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import q2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from q2.distributed import RankBatchSampler, accumulation_context, scaled_local_loss
from q2.train import EpochBatchSampler


def gradient_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).resolve().as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(12)
        # 22 samples: final global microbatch and accumulation group are partial.
        x = torch.arange(66, dtype=torch.float32).reshape(22, 3) / 31
        y = torch.arange(22) % 5
        raw = nn.Linear(3, 5)
        reference = nn.Linear(3, 5)
        reference.load_state_dict(raw.state_dict())
        model = DDP(raw)
        optimizer = torch.optim.AdamW(raw.parameters(), lr=.003, weight_decay=.01)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=.01)
        local_batches = list(RankBatchSampler(22, 6, 42, 0, rank))
        global_batches = list(EpochBatchSampler(22, 6, 42, 0))
        optimizer.zero_grad(); ref_optimizer.zero_grad()
        errors = []
        for index, indices in enumerate(local_batches):
            samples = min(12, 22 - (index // 2) * 12)
            boundary = (index + 1) % 2 == 0 or index + 1 == len(local_batches)
            with accumulation_context(model, boundary):
                loss = nn.functional.cross_entropy(model(x[indices]), y[indices], reduction='sum')
                scaled_local_loss(loss, samples, 2).backward()
            global_indices = global_batches[index]
            ref_loss = nn.functional.cross_entropy(reference(x[global_indices]), y[global_indices], reduction='sum')
            (ref_loss / samples).backward()
            if boundary:
                for a,b in zip(raw.parameters(), reference.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=2e-6, atol=2e-7)
                    errors.append(float((a.grad-b.grad).abs().max()))
                nn.utils.clip_grad_norm_(raw.parameters(),1)
                nn.utils.clip_grad_norm_(reference.parameters(),1)
                optimizer.step(); ref_optimizer.step()
                optimizer.zero_grad(); ref_optimizer.zero_grad()
        for a,b in zip(raw.parameters(), reference.parameters()):
            torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)
        if rank == 0:
            torch.save({'maximum_gradient_error':max(errors), 'status':'PASS'}, Path(output)/'gradients.pt')
    finally:
        dist.destroy_process_group()


def engine_worker(rank, rendezvous, prepared, output, config_path, scenario):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2',
                      Q2_DDP_TEST_INIT_METHOD=Path(rendezvous).resolve().as_uri()+f'?rank={rank}&world_size=2')
    from q2.common import load_config
    from q2.models import make_model
    from q2.train_distributed import run_distributed
    model_name = 'roberta_first' if scenario == 'first' else 'roberta_mean'
    def interrupting_model(*args, **kwargs):
        model = make_model(*args, **kwargs)
        forward = model.forward
        def wrapped(*items, **options):
            result = forward(*items, **options)
            if rank == 0:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return result
        model.forward = wrapped
        return model
    context = patch('q2.train_distributed.make_model', side_effect=interrupting_model) if scenario == 'interrupt' else nullcontext()
    with context:
        result = run_distributed(load_config(config_path), prepared, output, model_name, 42, allow_cpu=True)
    expected = 'interrupted' if scenario == 'interrupt' else 'complete'
    assert result['status'] == expected, result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',required=True)
    parser.add_argument('--rendezvous',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepared')
    parser.add_argument('--config')
    args=parser.parse_args()
    if args.scenario=='gradients':
        mp.spawn(gradient_worker,args=(args.rendezvous,args.output),nprocs=2,join=True)
    else:
        mp.spawn(engine_worker,args=(args.rendezvous,args.prepared,args.output,args.config,args.scenario),nprocs=2,join=True)
