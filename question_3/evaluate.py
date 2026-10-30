"""Final, reference-free greedy generation and full-test stemmed ROUGE F1."""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

from data import NewsDataset, collate, read_rows
from model import Transformer
from verify_inputs import verify


def sentences(text):
    # Offline sentence heuristic, applied identically to both system outputs.
    # Reference highlights retain their original newline sentence boundaries.
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text.strip()) if s.strip()]


def summary_text(ids, tokenizer):
    ids = list(ids)
    if tokenizer.eos_id() in ids:
        ids = ids[:ids.index(tokenizer.eos_id())]
    return '\n'.join(sentences(tokenizer.decode(ids)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='../work')
    parser.add_argument('--output', default='results')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    work, output = Path(args.work), Path(args.output)
    verify(work)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(work / 'test.jsonl')
    if len(rows) != 11490:
        raise ValueError('Final evaluation requires all 11,490 test articles.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = spm.SentencePieceProcessor(model_file=str(work / 'tokenizer.model'))
    checkpoint = torch.load(work / 'best.pt', map_location=device, weights_only=True)
    model = Transformer(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # Test highlights are never supplied to the model or used to select settings.
    articles = [dict(article=row['article'], highlights='') for row in rows]
    dataset = NewsDataset(articles, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True, split_summaries=False)
    totals = {system: np.zeros(4) for system in ('transformer', 'lead3')}
    count, examples = 0, []
    unique_predictions = set()
    started = time.perf_counter()
    ended = repeated = 0
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for source, _, _ in loader:
            generated, _ = model.generate(source.to(device))
            for ids in generated.cpu().tolist():
                row = rows[count]
                prediction = summary_text(ids, tokenizer)
                unique_predictions.add(prediction)
                lead3 = '\n'.join(sentences(row['article'])[:3])
                for system, text in [('transformer', prediction), ('lead3', lead3)]:
                    scores = scorer.score(row['highlights'], text)
                    totals[system] += [scores[key].fmeasure for key in metrics]
                ended += int(tokenizer.eos_id() in ids)
                words = prediction.split()
                trigrams = list(zip(words, words[1:], words[2:]))
                repeated += int(len(set(trigrams)) < len(trigrams))
                stream.write(json.dumps(dict(id=row['id'], prediction=prediction), ensure_ascii=False) + '\n')
                if count < 5:
                    # Short excerpts are enough to inspect facts; do not bundle test data.
                    examples.append(dict(index=count, id=row['id'],
                                         article_excerpt=row['article'][:1100],
                                         reference=row['highlights'], generated=prediction))
                count += 1
            if count % 320 == 0 or count == len(rows):
                print(f'Evaluated {count}/{len(rows)} ({time.perf_counter()-started:.0f}s)', flush=True)
    scores = {system: dict(zip(metrics, (100 * values / count).tolist()))
              for system, values in totals.items()}
    results = dict(test_pairs=count, scores=scores, unique_predictions=len(unique_predictions),
                   scale='0-100, macro mean per-example F1',
                   use_stemmer=True, best_epoch=checkpoint['epoch'],
                   decoding='greedy, 128 generated-token limit, cached encoder and decoder K/V',
                   eos_terminated=ended, repeated_trigram_examples=repeated,
                   evaluation_seconds=time.perf_counter()-started)
    (output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (output / 'examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    # Capture attention from the original generation path, not teacher forcing.
    source = dataset[0][0].unsqueeze(0).to(device)
    generated, attention = model.generate(source, capture=True)
    ids = generated[0].cpu().tolist()
    length = ids.index(tokenizer.eos_id()) if tokenizer.eos_id() in ids else len(ids)
    length = max(1, length)
    np.savez(output / 'attention.npz', weights=attention[0, :length].numpy(),
             source=np.array([tokenizer.id_to_piece(i) for i in source[0].cpu().tolist()]),
             target=np.array([tokenizer.id_to_piece(i) for i in ids[:length]]))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
