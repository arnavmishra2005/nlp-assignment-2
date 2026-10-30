"""Offline diagnostics on training/validation only; never update model weights."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sentencepiece import sentencepiece_model_pb2

from data import NewsDataset, collate, read_rows, train_tokenizer
from model import Transformer
from train import epoch_loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def uncached(model, source, limit=128):
    memory, mask = model.encode(source)
    prefix = torch.full((len(source), 1), 2, dtype=torch.long)
    done = torch.zeros(len(source), dtype=torch.bool)
    outputs = []
    for _ in range(limit):
        logits = model.decode(prefix, memory, mask)[0][:, -1]
        logits[:, [0, 2]] = float('-inf')
        token = logits.argmax(-1).masked_fill(done, 3)
        outputs.append(token)
        prefix = torch.cat([prefix, token[:, None]], dim=1)
        done |= token.eq(3)
        if done.all():
            break
    return torch.stack(outputs, dim=1)


def rms(x):
    return float(x.float().square().mean().sqrt())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='../work')
    parser.add_argument('--output', default='results/audit')
    parser.add_argument('--checkpoint', default='best.pt')
    parser.add_argument('--lr-scale', type=float, default=1.0)
    args = parser.parse_args()
    work, output = Path(args.work), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       cuda=torch.cuda.is_available(), torch_runtime=torch.__version__,
                       torch_file=torch.__file__, packages={})
    for name, module in [('torch', 'torch'), ('numpy', 'numpy'), ('datasets', 'datasets'),
                         ('sentencepiece', 'sentencepiece'), ('rouge-score', 'rouge_score'),
                         ('nltk', 'nltk'), ('matplotlib', 'matplotlib'), ('reportlab', 'reportlab'),
                         ('protobuf', 'google.protobuf')]:
        imported = importlib.import_module(module)
        environment['packages'][name] = dict(version=metadata.version(name), file=imported.__file__)
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    train_rows, val_rows = read_rows(work / 'train.jsonl'), read_rows(work / 'validation.jsonl')
    tokenizer = spm.SentencePieceProcessor(model_file=str(work / 'tokenizer.model'))
    checkpoint_hash = digest(work / args.checkpoint)
    checkpoint = torch.load(work / args.checkpoint, weights_only=True, map_location='cpu')
    model = Transformer(**checkpoint['config']).eval()
    model.load_state_dict(checkpoint['model'])
    indices = [0, 1, 2, 100, 500, 1000]
    dataset = NewsDataset([val_rows[i] for i in indices], tokenizer)
    source, target, labels = collate([dataset[i] for i in range(len(dataset))])
    with torch.inference_mode():
        memory, mask = model.encode(source)
        layers = []
        x = model.embed(source)
        for layer in model.encoder:
            x = layer(x, mask)
            layers.append(dict(pairwise_rms_vs_first=[rms(x[i]-x[0]) for i in range(1, len(x))],
                               token_std=float(x.std(dim=1).mean())))
        logits = model.decode(target, memory, mask)[0]
        losses = {}
        for name, mem in [('correct', memory), ('shuffled', memory.flip(0)),
                          ('zero', torch.zeros_like(memory))]:
            pred = model.decode(target, mem, mask)[0]
            losses[name] = dict(loss=float(F.cross_entropy(pred.flatten(0, 1), labels.flatten(),
                                                          ignore_index=0)),
                                max_logit_change=float((pred-logits).abs().max()))
        # Verify projected encoder K/V and the exact source padding mask at every step.
        routing = []
        for layer in model.decoder:
            expected = layer.cross_attention.project_kv(memory)
            original = layer.cross_attention.attend
            records = []

            def checked(query, key, value, blocked=None, original=original,
                        expected=expected, records=records):
                torch.testing.assert_close(key, expected[0])
                torch.testing.assert_close(value, expected[1])
                assert torch.equal(blocked, mask)
                result, weights = original(query, key, value, blocked)
                assert torch.all(weights.masked_select(mask.expand_as(weights)) == 0)
                records.append(True)
                return result, weights

            routing.append((layer.cross_attention, checked, records))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for attention, checked, _ in routing:
                stack.enter_context(patch.object(attention, 'attend', side_effect=checked))
            cached, attention = model.generate(source, capture=True)
        full = uncached(model, source)
        assert torch.equal(cached, full), 'Cached and full-prefix generation differ'
        # Repeat per article: catches row broadcasting and padding/batch dependence.
        individual = []
        for i in range(len(dataset)):
            single, _ = model.generate(dataset[i][0][None])
            end = cached[i].tolist().index(3) + 1 if 3 in cached[i] else cached.size(1)
            assert torch.equal(single[0, :end], cached[i, :end])
            individual.append(True)
        # Real padding is exercised by an additional short VALIDATION source prefix.
        padded = collate([dataset[0], (dataset[1][0][:37], dataset[1][1])])[0]
        pmem, pmask = model.encode(padded)
        _, pw = model.decode(torch.full((2, 1), 2), pmem, pmask)
        assert torch.all(pw.masked_select(pmask.expand_as(pw)) == 0)
        assert torch.equal(model.generate(padded)[0], uncached(model, padded))
        # Check cached/full-prefix logits directly along a fixed validation gold prefix.
        caches = [None] * len(model.decoder)
        kv = [layer.cross_attention.project_kv(memory) for layer in model.decoder]
        max_error = 0
        for position in range(20):
            x = model.embed(target[:, position:position+1], position)
            for j, layer in enumerate(model.decoder):
                x, _, caches[j] = layer.step(x, kv[j], mask, caches[j])
            actual = model.output(x[:, -1])
            expected = model.decode(target[:, :position+1], memory, mask)[0][:, -1]
            max_error = max(max_error, float((actual-expected).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        # Fresh random weights should exhibit source-dependent representations/logits.
        torch.manual_seed(42)
        fresh = Transformer(**checkpoint['config']).eval()
        fresh_memory, _ = fresh.encode(source)
        fresh_logits = fresh(source, torch.full((len(source), 1), 2))
        first_logits = model(source, torch.full((len(source), 1), 2))
    val_loss, _ = epoch_loss(model, DataLoader(NewsDataset(val_rows, tokenizer),
                             batch_size=16, collate_fn=collate), 'cpu')
    assert abs(val_loss-checkpoint['validation_loss']) < 1e-5
    # Diagnose gradients without changing checkpoint/model parameters.
    model.eval()
    loss = F.cross_entropy(model(source, target).flatten(0, 1), labels.flatten(), ignore_index=0)
    loss.backward()
    gradients = {}
    for prefix in ['embedding', 'encoder', 'decoder', 'output']:
        values = [p.grad for name, p in model.named_parameters()
                  if name.startswith(prefix) and p.grad is not None]
        assert all(torch.isfinite(g).all() for g in values)
        gradients[prefix] = sum(float(g.square().sum()) for g in values) ** 0.5
    # Rebuild the tokenizer from the original TRAINING rows to audit provenance.
    rebuild = work / 'audit_tokenizer'
    rebuild.mkdir(exist_ok=True)
    train_tokenizer(train_rows, rebuild)
    old = sentencepiece_model_pb2.ModelProto()
    new = sentencepiece_model_pb2.ModelProto()
    old.ParseFromString((work / 'tokenizer.model').read_bytes())
    new.ParseFromString((rebuild / 'tokenizer.model').read_bytes())
    assert old.pieces == new.pieces and old.normalizer_spec == new.normalizer_spec
    assert old.trainer_spec.model_type == 2 and tokenizer.vocab_size() == 4096
    assert [tokenizer.pad_id(), tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()] == [0,1,2,3]
    assert digest(work / 'tokenizer_training.txt') == digest(rebuild / 'tokenizer_training.txt')
    assert not ({r['id'] for r in train_rows} & {r['id'] for r in val_rows})
    assert digest(work / args.checkpoint) == checkpoint_hash
    result = dict(validation_indices=indices, validation_ids=[val_rows[i]['id'] for i in indices],
                  unique_sources=len({tuple(item[0].tolist()) for item in dataset}),
                  source_token_hashes=[hashlib.sha256(item[0].numpy().tobytes()).hexdigest() for item in dataset],
                  source_lengths=[len(item[0]) for item in dataset],
                  trained_encoder_layers=layers,
                  trained_memory_rms_vs_first=[rms(memory[i]-memory[0]) for i in range(1,len(source))],
                  fresh_memory_rms_vs_first=[rms(fresh_memory[i]-fresh_memory[0]) for i in range(1,len(source))],
                  trained_first_logits_rms_vs_first=[rms(first_logits[i]-first_logits[0]) for i in range(1,len(source))],
                  fresh_first_logits_rms_vs_first=[rms(fresh_logits[i]-fresh_logits[0]) for i in range(1,len(source))],
                  memory_ablation=losses, cached_equals_uncached=True,
                  max_cached_vs_full_logit_error=max_error, individual_matches_batch=individual,
                  cross_memory_and_mask_checked_steps=[len(r[2]) for r in routing],
                  real_padding_check=True, attention_min=float(attention.min()),
                  attention_max=float(attention.max()), gradient_l2=gradients,
                  validation_loss_recomputed=val_loss, checkpoint_validation_loss=checkpoint['validation_loss'],
                  unique_validation_predictions=len({tuple(x) for x in cached.tolist()}),
                  validation_predictions=[tokenizer.decode(row[:row.index(3)] if 3 in row else row)
                                          for row in cached.tolist()],
                  tokenizer_rebuild_identical_pieces_and_scores=True,
                  tokenizer_corpus_hash=digest(rebuild / 'tokenizer_training.txt'),
                  tokenizer_sha256=digest(work / 'tokenizer.model'), checkpoint_sha256=checkpoint_hash,
                  schedule=[dict(step=s, lr=args.lr_scale*64**-0.5*min(s**-0.5,s*200**-1.5))
                            for s in [1,128,200,256,384,512,640]],
                  model_updated=False, test_data_used=False)
    (output / 'diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
