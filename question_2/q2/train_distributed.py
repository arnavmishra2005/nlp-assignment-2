"""Two-process RoBERTa training with the original global batches and outputs."""
import argparse
from datetime import timedelta
import gc
import logging
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .common import (atomic_torch_save, capture_dependencies, digest, environment,
                     file_hash, load_config, logger, read_json, seed_everything,
                     source_hash, write_json)
from .data import load_training_data, verify_prepared
from .distributed import (RankBatchSampler, accumulation_context, check_gradient_equivalence, local_rng_state,
                          rank_records, reduced_statistics, restore_local_rng,
                          scaled_local_loss, topology)
from .metrics import metrics_from_confusion
from .models import make_model
from .train import (autocast_context, collator_for, evaluate_loader, move_batch,
                    precision_settings, update_confusion)
from .validation import verify_completed_run


def run_distributed(config, prepared, output, model_name, seed, resume=True, allow_cpu=False):
    if model_name not in ('roberta_mean', 'roberta_first'):
        raise ValueError('Only the remaining RoBERTa experiments use DDP; baseline trainer is unchanged')
    if seed not in config['seeds']:
        raise ValueError('Seed not in the frozen experiment plan')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    execution = topology(config, world)
    if allow_cpu:
        device, backend = torch.device('cpu'), 'gloo'
        torch.set_num_threads(1)
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError('DUAL_GPU_REQUIRED: select GPU T4 x2')
        torch.cuda.set_device(local_rank)
        device, backend = torch.device('cuda', local_rank), 'nccl'
    init_method = os.environ.get('Q2_DDP_TEST_INIT_METHOD', 'env://') if allow_cpu else 'env://'
    dist.init_process_group(backend=backend, init_method=init_method, timeout=timedelta(minutes=30))
    try:
        return train_worker(config, Path(prepared), Path(output), model_name, seed,
                            resume, rank, world, device, execution)
    finally:
        dist.destroy_process_group()


def train_worker(config, prepared, output, model_name, seed, resume, rank, world, device, execution):
    # Prove both processes can communicate before loading/training the large model.
    probe = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(probe)
    if probe.item() != 3.0:
        raise RuntimeError('Two-rank all-reduce failed')
    devices = rank_records({'rank': rank, 'device': str(device), 'pid': os.getpid(),
                            'name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU test',
                            'all_reduce_checksum': probe.item()}, world)
    gradient_probe = check_gradient_equivalence(device)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    if rank == 0:
        log = logger(output / 'training.log')
        log.propagate = False
        log.info('DUAL_GPU_COMMUNICATION_PASSED | ranks=%s', devices)
        log.info('DDP_GRADIENT_CHECK_PASSED | %s', gradient_probe)
        log.info('DDP_TRAINING_ENABLED | %s', execution)
    else:
        log = logging.getLogger(f'q2.rank{rank}')
        log.handlers = [logging.NullHandler()]
        log.propagate = False
    manifest = verify_prepared(prepared, config)
    signature = digest({'config': config, 'prepared_manifest': digest(manifest), 'model': model_name,
                        'seed': seed, 'source': source_hash(), 'distributed': execution})
    run_file = output / 'run.json'
    if run_file.exists() and read_json(run_file)['signature'] != signature:
        raise ValueError('Different code/config/data/topology in output directory; start a fresh session')
    if (output / 'complete.json').exists():
        verify_completed_run(output, expected_signature=signature)
        return read_json(output / 'complete.json')
    seed_everything(seed)
    c, t = config['roberta'], config['training']
    dtype = precision_settings(c['precision'], device)
    train_data, val_data = load_training_data(prepared, model_name)
    expected = config['dataset']['expected_sizes']
    if len(train_data) != expected['train'] or len(val_data) != expected['validation']:
        raise ValueError('Wrong official train/validation counts; refusing subset training')
    # Validate divisibility before DDP forward calls. No DistributedSampler padding.
    RankBatchSampler(len(train_data), c['batch_size'], seed, 0, rank, world)
    collator = collator_for(model_name, prepared)
    raw_model = make_model(model_name, config, prepared=prepared).to(device)
    metadata = {'encoder_config': raw_model.encoder.config.to_dict()}
    parameters = sum(p.numel() for p in raw_model.parameters())
    batches_per_epoch = math.ceil(len(train_data) / c['batch_size'])
    updates_per_epoch = math.ceil(batches_per_epoch / c['accumulation_steps'])
    # Same parameter groups, AdamW settings and scheduler update counts as train.py.
    decay, no_decay = [], []
    for name, parameter in raw_model.named_parameters():
        (no_decay if name.endswith('bias') or 'LayerNorm.weight' in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': c['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0.0}], lr=c['learning_rate'])
    from transformers import get_linear_schedule_with_warmup
    total_updates = updates_per_epoch * c['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_updates * c['warmup_ratio']), total_updates)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    model = DDP(raw_model, device_ids=[device.index] if device.type == 'cuda' else None,
                broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
    # Same initial pretrained model/head seed; independent, recorded dropout streams.
    seed_everything(seed + rank)
    run = {'signature': signature, 'config': config, 'seed': seed, 'model': model_name,
           'metadata': metadata, 'parameters': parameters,
           'prepared_manifest': digest(manifest), 'precision': str(dtype or torch.float32),
           'planned_optimizer_updates': total_updates, 'source_hash': source_hash(),
           'distributed': {**execution, 'backend': dist.get_backend(), 'devices': devices,
                           'gradient_probe': gradient_probe,
                           'dropout_seeds': [seed + index for index in range(world)],
                           'validation': 'rank0, all official validation rows, original evaluation batch size'}}
    if rank == 0:
        run['environment'] = environment()
        prior = read_json(run_file) if run_file.exists() else None
        run['sessions'] = (prior.get('sessions', [prior['environment']]) if prior else []) + [run['environment']]
        write_json(run_file, run)
        capture_dependencies(output)
        write_json(output / 'distributed_execution.json', run['distributed'])
    dist.barrier()
    # All ranks save/reuse identical run metadata; only rank0 writes files.
    run = read_json(run_file)
    state = {'epoch': 0, 'next_batch': 0, 'optimizer_steps': 0, 'best_f1': -1.0,
             'best_epoch': None, 'bad_epochs': 0, 'history': [], 'elapsed_seconds': 0.0,
             'partial': {'loss_sum': 0.0, 'count': 0, 'confusion': np.zeros((5, 5), dtype=np.int64)}}
    last = output / 'last.pt'
    if last.exists():
        if not resume:
            raise ValueError('Recovery exists; resume or use fresh output')
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        if checkpoint['signature'] != signature or checkpoint['distributed'] != execution:
            raise ValueError('Recovery source/config/world-size mismatch')
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        state = checkpoint['state']
        state['partial'] = checkpoint['rank_states'][rank]['partial']
        restore_local_rng(checkpoint['rank_states'][rank]['rng'], device)
        del checkpoint
        log.info('DDP RESUME epoch=%d next global microbatch=%d', state['epoch'] + 1, state['next_batch'])
    log.info('START model=%s seed=%d train=%d validation=%d max_epochs=%d | local batch=%d x %d GPUs x accumulation=%d = effective batch=%d | updates/epoch=%d',
             model_name, seed, len(train_data), len(val_data), c['epochs'], execution['per_gpu_microbatch'], world,
             c['accumulation_steps'], execution['effective_batch_size'], updates_per_epoch)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    session_start, elapsed_before = time.monotonic(), state['elapsed_seconds']
    stop_requested = [False]
    previous_handler = signal.signal(signal.SIGTERM, lambda signum, frame: stop_requested.__setitem__(0, True))

    def save_recovery():
        state['elapsed_seconds'] = elapsed_before + time.monotonic() - session_start
        records = rank_records({'partial': state['partial'], 'rng': local_rng_state(device)}, world)
        if rank == 0:
            atomic_torch_save({'signature': signature, 'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'state': state, 'rank_states': records,
                'distributed': execution}, last)
            write_json(output / 'status.json', {'status': 'training', 'epoch': state['epoch'] + 1,
                'next_batch': state['next_batch'], 'optimizer_steps': state['optimizer_steps'],
                'best_validation_macro_f1': state['best_f1'], 'elapsed_seconds': state['elapsed_seconds']})
            log.info('Recovery saved: epoch=%d next global microbatch=%d (both rank RNG/partial states)', state['epoch'] + 1, state['next_batch'])
        dist.barrier()

    try:
        while state['epoch'] < c['epochs'] and state['bad_epochs'] < c['patience']:
            epoch = state['epoch']
            epoch_start = time.monotonic()
            epoch_elapsed_before = state['partial'].get('elapsed_seconds', 0.0)
            sampler = RankBatchSampler(len(train_data), c['batch_size'], seed, epoch, rank, world, state['next_batch'])
            loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collator,
                num_workers=t['num_workers'], pin_memory=device.type == 'cuda',
                multiprocessing_context='spawn' if t['num_workers'] else None,
                generator=torch.Generator().manual_seed(seed + epoch))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            partial = state['partial']
            for index, batch in enumerate(loader, start=state['next_batch']):
                batch = move_batch(batch, device)
                labels = batch.pop('labels')
                group_begin = (index // c['accumulation_steps']) * c['accumulation_steps'] * c['batch_size']
                group_samples = min(c['accumulation_steps'] * c['batch_size'], len(train_data) - group_begin)
                boundary = (index + 1) % c['accumulation_steps'] == 0 or index + 1 == batches_per_epoch
                with accumulation_context(model, boundary):
                    with autocast_context(device, dtype):
                        logits = model(**batch)
                        batch_loss = nn.functional.cross_entropy(logits.float(), labels, reduction='sum')
                        loss = scaled_local_loss(batch_loss, group_samples, world)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss on rank {rank}')
                    scaler.scale(loss).backward()
                partial['loss_sum'] += batch_loss.item()
                partial['count'] += len(labels)
                update_confusion(partial['confusion'], labels, logits.argmax(-1))
                if boundary:
                    scaler.unscale_(optimizer)
                    norm = nn.utils.clip_grad_norm_(raw_model.parameters(), t['gradient_clip'])
                    if not torch.isfinite(norm) and not scaler.is_enabled():
                        raise FloatingPointError('Nonfinite gradients')
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    update_applied = scaler.get_scale() >= scale_before
                    if update_applied:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    state['optimizer_steps'] += int(update_applied)
                    state['next_batch'] = index + 1
                    stopping = torch.tensor(int(stop_requested[0]), device=device)
                    dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
                    if stopping.item() or (update_applied and state['optimizer_steps'] % t['checkpoint_every_optimizer_steps'] == 0):
                        partial['elapsed_seconds'] = epoch_elapsed_before + time.monotonic() - epoch_start
                        save_recovery()
                    if stopping.item():
                        if rank == 0:
                            write_json(output / 'status.json', {'status': 'interrupted', 'next_batch': index + 1})
                        return {'status': 'interrupted'}
                if (index + 1) % t['log_every_batches'] == 0 or index + 1 == batches_per_epoch:
                    global_partial = reduced_statistics(partial, device)
                    running = metrics_from_confusion(global_partial['confusion'])
                    memory = rank_records(torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0, world)
                    log.info('epoch=%d global batch=%d/%d loss=%.4f train F1=%.4f lr=%.3g elapsed=%.1f min | GPUs=%d local_batch=%d | peak GiB/rank=%s',
                        epoch + 1, index + 1, batches_per_epoch, global_partial['loss_sum'] / global_partial['count'],
                        running['macro_f1'], optimizer.param_groups[0]['lr'], (time.monotonic() - session_start)/60,
                        world, execution['per_gpu_microbatch'], memory)
            global_partial = reduced_statistics(partial, device)
            if global_partial['count'] != len(train_data):
                raise ValueError('Distributed epoch did not cover every training record exactly once')
            validation_holder = [None]
            if rank == 0:
                loader = DataLoader(val_data, batch_size=c['eval_batch_size'], shuffle=False, collate_fn=collator,
                    num_workers=t['num_workers'], pin_memory=device.type == 'cuda',
                    multiprocessing_context='spawn' if t['num_workers'] else None,
                    generator=torch.Generator().manual_seed(seed))
                # Unwrapped model avoids collectives while the other rank waits.
                validation_holder[0] = evaluate_loader(raw_model, loader, device, dtype)
            dist.broadcast_object_list(validation_holder, src=0)
            validation = validation_holder[0]
            training = metrics_from_confusion(global_partial['confusion'])
            training['loss'] = global_partial['loss_sum'] / global_partial['count']
            epoch_seconds = epoch_elapsed_before + time.monotonic() - epoch_start
            elapsed_total = elapsed_before + time.monotonic() - session_start
            row = {'epoch': epoch + 1, 'train': training, 'validation': validation, 'epoch_seconds': epoch_seconds,
                   'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_seconds': elapsed_total}
            state['history'].append(row)
            if validation['macro_f1'] > state['best_f1'] + t['min_delta']:
                state['best_f1'], state['best_epoch'], state['bad_epochs'] = validation['macro_f1'], epoch + 1, 0
                if rank == 0:
                    atomic_torch_save({'signature': signature, 'model': raw_model.state_dict(), 'run': run,
                        'best_epoch': epoch + 1, 'validation': validation}, output / 'best.pt')
            else:
                state['bad_epochs'] += 1
            if rank == 0:
                write_json(output / 'history.json', state['history'])
            log.info('EPOCH %d COMPLETE | train loss=%.4f F1=%.4f | val loss=%.4f F1=%.4f | best epoch=%d | patience=%d/%d',
                epoch + 1, training['loss'], training['macro_f1'], validation['loss'], validation['macro_f1'],
                state['best_epoch'], state['bad_epochs'], c['patience'])
            log.info('[progress] epoch_seconds=%.1f approximate remaining at epoch cap=%.1f min; early stopping may shorten',
                epoch_seconds, epoch_seconds * (c['epochs'] - epoch - 1) / 60)
            state['epoch'], state['next_batch'] = epoch + 1, 0
            state['partial'] = {'loss_sum': 0.0, 'count': 0, 'confusion': np.zeros((5, 5), dtype=np.int64)}
            save_recovery()
        memory = rank_records(torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0, world)
        result_holder = [None]
        if rank == 0:
            result = {'status': 'complete', 'signature': signature, 'model': model_name, 'seed': seed,
                'best_epoch': state['best_epoch'], 'best_validation_macro_f1': state['best_f1'],
                'epochs_completed': state['epoch'], 'elapsed_seconds': state['elapsed_seconds'],
                'best_sha256': file_hash(output / 'best.pt'), 'parameters': parameters,
                'peak_gpu_memory_gib': max(memory), 'per_rank_peak_gpu_memory_gib': memory,
                'distributed': execution}
            result['evidence_sha256'] = {name: file_hash(output / name) for name in
                ('run.json', 'history.json', 'best.pt', 'environment-freeze.txt', 'distributed_execution.json')}
            write_json(output / 'complete.json', result)
            write_json(output / 'status.json', result)
            result_holder[0] = result
            log.info('TRAINING COMPLETE | %s seed %d | %.2f hours | best val F1=%.4f | both ranks active | test untouched',
                model_name, seed, result['elapsed_seconds'] / 3600, state['best_f1'])
        dist.broadcast_object_list(result_holder, src=0)
        return result_holder[0]
    except Exception:
        if rank == 0:
            log.exception('DDP RUN FAILED; keep both-rank recovery state and logs')
            write_json(output / 'failure.json', {'environment': environment(), 'recovery_available': last.exists()})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        del model, raw_model, optimizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['roberta_mean', 'roberta_first'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--allow-cpu', action='store_true', help='Developer two-process Gloo tests only')
    args = parser.parse_args()
    result = run_distributed(load_config(args.config), args.prepared, args.output, args.model,
                             args.seed, not args.no_resume, args.allow_cpu)
    if result['status'] != 'complete':
        raise SystemExit('Interrupted, recovery saved; experiment is not complete')


if __name__ == '__main__':
    main()
