"""Record actual training topology rather than the number of allocated devices."""


def execution_details(run):
    config = run['config']['baseline' if run['model'] == 'baseline' else 'roberta']
    distributed = run.get('distributed') or {}
    world = distributed.get('world_size', 1)
    return {'strategy': distributed.get('strategy', 'single_process'),
            'world_size': world,
            'per_device_microbatch': distributed.get('per_gpu_microbatch', config['batch_size']),
            'global_microbatch': config['batch_size'],
            'accumulation_steps': config['accumulation_steps'],
            'effective_batch_size': config['batch_size'] * config['accumulation_steps'],
            'source_hash': run['source_hash']}


def execution_note(details):
    transformer = [row for row in details if row['model'].startswith('roberta_')]
    if len({(row['strategy'], row['world_size']) for row in transformer}) > 1:
        return ('**Execution limitation:** RoBERTa runs used a mixture of single-process and '
                'two-GPU DDP training, recorded per model/seed below. The completed mean-pooling '
                'seed42 run was retained; subsequent runs use two GPUs with the same global '
                'microbatch, effective batch, data, optimizer and selection settings. Per-rank '
                'dropout streams and floating-point reductions differ. The pooling comparison '
                'is therefore not fully controlled for execution topology; do not attribute all '
                'observed differences to pooling or interpret the two-seed spread as seed-only variability.')
    return ('Training execution is recorded per model/seed below. Visible GPU allocation alone '
            'does not establish how many devices trained the model. Timing comparisons must account for topology.')
