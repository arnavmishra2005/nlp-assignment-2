"""Two-rank execution helpers preserving the original global minibatches."""
from contextlib import nullcontext
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from .train import EpochBatchSampler


def topology(config, world_size=2):
    c = config['roberta']
    if world_size != 2 or c['batch_size'] % world_size:
        raise ValueError('This workflow requires two ranks and an evenly divisible global microbatch')
    return {'strategy': 'DistributedDataParallel', 'world_size': world_size,
            'per_gpu_microbatch': c['batch_size'] // world_size,
            'global_microbatch': c['batch_size'], 'accumulation_steps': c['accumulation_steps'],
            'effective_batch_size': c['batch_size'] * c['accumulation_steps'],
            'sampler': 'original global seed/epoch permutation, disjoint contiguous rank slices',
            'gradient_reduction': 'world_size * local summed loss / actual global accumulation-group samples'}


class RankBatchSampler(EpochBatchSampler):
    """Union of rank slices is exactly each original minibatch; no padded samples."""
    def __init__(self, size, batch_size, seed, epoch, rank, world_size=2, start_batch=0):
        super().__init__(size, batch_size, seed, epoch, start_batch)
        if not 0 <= rank < world_size or batch_size % world_size or size % world_size:
            raise ValueError('Rank/global batch/dataset must permit equal nonempty rank slices; no duplication allowed')
        self.rank, self.world_size = rank, world_size

    def __iter__(self):
        for batch in super().__iter__():
            local = len(batch) // self.world_size
            yield batch[self.rank * local:(self.rank + 1) * local]


def accumulation_context(model, boundary):
    # no_sync must cover the forward AND backward pass.
    return nullcontext() if boundary else model.no_sync()


def scaled_local_loss(local_sum, group_samples, world_size):
    # DDP averages rank gradients. Undo that averaging before dividing by the
    # same actual group size as the original single-GPU trainer.
    return local_sum * world_size / group_samples


def reduced_statistics(partial, device):
    packed = torch.tensor([partial['loss_sum'], partial['count'],
                           *np.asarray(partial['confusion']).reshape(-1)],
                          dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.cpu().numpy()
    return {'loss_sum': float(values[0]), 'count': int(round(values[1])),
            'confusion': np.rint(values[2:]).astype(np.int64).reshape(5, 5)}


def local_rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_local_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def rank_records(record, world_size=2):
    records = [None] * world_size
    dist.all_gather_object(records, record)
    return records


def check_gradient_equivalence(device):
    """Tiny two-rank arithmetic check before any assignment-model optimizer step.

    No real reviews are loaded. The real experiment resets its seeds afterward.
    This checks the target backend, no_sync, partial groups, clipping and AdamW.
    """
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(12)
    x = torch.arange(66, dtype=torch.float32, device=device).reshape(22, 3) / 31
    y = torch.arange(22, device=device) % 5
    raw = nn.Linear(3, 5).to(device)
    reference = nn.Linear(3, 5).to(device)
    reference.load_state_dict(raw.state_dict())
    model = DDP(raw, device_ids=[device.index] if device.type == 'cuda' else None)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=.003, weight_decay=.01)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=.01)
    local_batches = list(RankBatchSampler(22, 6, 42, 0, rank, world))
    global_batches = list(EpochBatchSampler(22, 6, 42, 0))
    optimizer.zero_grad(); ref_optimizer.zero_grad()
    max_error = 0.0
    for index, indices in enumerate(local_batches):
        samples = min(12, 22 - (index // 2) * 12)
        boundary = (index + 1) % 2 == 0 or index + 1 == len(local_batches)
        with accumulation_context(model, boundary):
            loss = nn.functional.cross_entropy(model(x[indices]), y[indices], reduction='sum')
            scaled_local_loss(loss, samples, world).backward()
        original = global_batches[index]
        ref_loss = nn.functional.cross_entropy(reference(x[original]), y[original], reduction='sum')
        (ref_loss / samples).backward()
        if boundary:
            for a, b in zip(raw.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)
                max_error = max(max_error, float((a.grad - b.grad).abs().max().item()))
            nn.utils.clip_grad_norm_(raw.parameters(), 1)
            nn.utils.clip_grad_norm_(reference.parameters(), 1)
            optimizer.step(); ref_optimizer.step()
            optimizer.zero_grad(); ref_optimizer.zero_grad()
    for a, b in zip(raw.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    error = torch.tensor(max_error, device=device)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    report = {'status': 'PASS', 'maximum_gradient_error': float(error.item()),
              'samples': 22, 'global_microbatch': 6, 'accumulation_steps': 2,
              'partial_final_group_tested': True, 'real_training_data_used': False}
    del model, raw, reference, optimizer, ref_optimizer
    return report


def check_dual_gpu(config, output=None):
    """Read-only device probes before launching worker processes; no training."""
    from .common import write_json
    plan = topology(config)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('DUAL_GPU_REQUIRED: select Kaggle GPU T4 x2; no silent single-GPU fallback')
    probes = []
    for index in range(2):
        with torch.cuda.device(index):
            x = torch.ones((4, 4), device=f'cuda:{index}')
            checksum = float((x @ x).sum().item())
            if checksum != 64:
                raise RuntimeError(f'GPU {index} operation failed')
            probes.append({'rank': index, 'device': f'cuda:{index}',
                           'name': torch.cuda.get_device_name(index), 'checksum': checksum})
            del x
            torch.cuda.empty_cache()
    report = {'status': 'PASS', 'topology': plan, 'devices': probes,
              'communication_check': 'NCCL all-reduce is checked by the two workers before optimizer execution'}
    if output is not None:
        write_json(output, report)
    print('DUAL_GPU_SETUP_PASSED', report, flush=True)
    return report
