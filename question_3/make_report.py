"""Build the final PDF and figures directly from measured experiment outputs."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name='Body', parent=STYLES['BodyText'], fontSize=10,
                         leading=14, spaceAfter=9))
STYLES.add(ParagraphStyle(name='SmallBody', parent=STYLES['Body'], fontSize=9, leading=12))


def paragraph(text, style='Body'):
    return Paragraph(escape(text).replace('\n', '<br/>'), STYLES[style])


def heading(text):
    return paragraph(text, 'Heading1')


def table(rows, widths=None):
    cells = [[paragraph(str(cell), 'SmallBody') for cell in row] for row in rows]
    result = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1)
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5edf4')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#bcc9d4')),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    return result


def figures():
    with (RESULTS / 'history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    fig, axis = plt.subplots(figsize=(8, 3.8))
    for key, label in [('train_loss', 'Training'), ('validation_loss', 'Validation')]:
        axis.plot([int(x['epoch']) for x in history], [float(x[key]) for x in history],
                  marker='o', label=label)
    axis.set(xlabel='Epoch', ylabel='Token-weighted cross-entropy (nats)',
             title='Teacher-forced loss; padding excluded')
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'loss_curves.png', dpi=180)
    plt.close(fig)
    with (RESULTS / 'audit/original_history.csv').open() as stream:
        old_history = list(csv.DictReader(stream))
    fig, axis = plt.subplots(figsize=(8, 3.8))
    axis.plot([int(x['epoch']) for x in old_history],
              [float(x['validation_loss']) for x in old_history], 'o--', label='Original: scale 1.0')
    axis.plot([int(x['epoch']) for x in history],
              [float(x['validation_loss']) for x in history], 'o-', label='Revised: scale 0.1')
    axis.set(xlabel='Epoch', ylabel='Validation cross-entropy (nats)',
             title='Controlled learning-rate revision; same model, data and seed')
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'stability_comparison.png', dpi=180)
    plt.close(fig)

    data = np.load(RESULTS / 'attention.npz')
    weights, source, target = data['weights'], data['source'], data['target']
    window = min(50, len(source))
    masses = np.convolve(weights.sum(0), np.ones(window), mode='valid')
    start = int(masses.argmax())
    # Retain original probabilities: cropping does not renormalize attention.
    shown = min(24, len(target))
    fig, axis = plt.subplots(figsize=(10, 6))
    im = axis.imshow(weights[:shown, start:start + window], aspect='auto', cmap='viridis')
    labels = lambda values: [str(x).replace('\u2581', '_') for x in values]
    axis.set_xticks(range(window), labels(source[start:start + window]), rotation=90, fontsize=10)
    axis.set_yticks(range(shown), labels(target[:shown]), fontsize=10)
    axis.set(xlabel=f'Source subwords (positions {start} to {start + window - 1}); _ marks word boundary',
             ylabel='Generated summary subwords', title='Last decoder layer cross-attention, mean of 4 heads')
    fig.colorbar(im, ax=axis, label='Attention probability')
    fig.tight_layout()
    fig.savefig(RESULTS / 'attention_heatmap.png', dpi=190)
    plt.close(fig)
    return history, dict(start=start, window=window, shown=shown,
                         mass=float(weights[:, start:start + window].sum(1).mean()),
                         max_weight=float(weights.max()),
                         row_sum_error=float(np.max(np.abs(weights.sum(1) - 1))))


def example_analysis(example):
    generated = example['generated']
    words = generated.split()
    triples = list(zip(words, words[1:], words[2:]))
    repeated = len(triples) - len(set(triples))
    reference_words = set(example['reference'].lower().split())
    overlap = len(set(words) & reference_words)
    facts = ['ICC jurisdiction and Palestinian membership',
             'Theia surviving severe injuries, her rescue and care',
             'Zarif, diplomacy with John Kerry, and English tweets',
             'Americans exposed to Ebola and the Maryland patient treatment status',
             'Duke student disciplinary review following the noose incident']
    anchors = [['icc', 'palestin'], ['theia', 'dog'], ['zarif', 'kerry'],
               ['ebola', 'maryland'], ['duke', 'noose']][example['index']]
    present = [word for word in anchors if word in generated.lower()]
    return (f"The key reference information is {facts[example['index']]}. "
            f"Of the diagnostic source-name/topic strings {', '.join(anchors)}, "
            f"the output contains {', '.join(present) if present else 'none'}. "
            'String presence alone does not establish a correct factual statement.\n\n' +
            f'The unedited output contains {len(words)} whitespace-delimited words; '
            f'{repeated} repeated trigram occurrences beyond their first appearances. '
            f'It shares {overlap} distinct whitespace tokens with the reference. '
            'These counts diagnose repetition and lexical overlap, not factual accuracy. '
            'Compare the source excerpt and reference above to check whether the output '
            'preserves the event, participants, and quantities. Generic or repeated text '
            'does not identify the key facts even when it contains common reference words.')


def footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#596b79'))
    canvas.drawString(44, 27, 'DSL 504 | Question 3 | Transformer summarization')
    canvas.drawRightString(551, 27, str(doc.page))


def main():
    training = json.loads((RESULTS / 'training.json').read_text())
    metrics = json.loads((RESULTS / 'metrics.json').read_text())
    examples = json.loads((RESULTS / 'examples.json').read_text())
    with (RESULTS / 'predictions.jsonl').open(encoding='utf-8') as stream:
        unique_predictions = len({json.loads(line)['prediction'] for line in stream})
    history, attention = figures()
    shown_history = [row for i, row in enumerate(history)
                     if i % 2 == 0 or i == len(history)-1
                     or int(row['epoch']) == training['best_epoch']]
    story = [paragraph('Abstractive Text Summarization', 'Title'),
             paragraph('Question 3 - From-scratch Transformer Encoder-Decoder', 'Heading2'),
             paragraph('Measured CPU experiment on CNN / Daily Mail 3.0.0. '
                       'All reported model scores come from autoregressive generation with the '
                       'validation-selected checkpoint; no pre-trained model weights are used.'),
             heading('1. Data and preprocessing'),
             paragraph('The official splits contain 287,113 training, 13,368 validation, and 11,490 test '
                       'pairs. To fit CPU resources, this experiment uses the first 2,048 training pairs '
                       'and first 1,024 validation pairs in repository order. The training subset is '
                       '0.71% of the full training split and is not a random representative sample. '
                       'All 11,490 test articles are evaluated. No test example trains the tokenizer, '
                       'model, or hyperparameters.'),
             paragraph('SentencePiece BPE is trained exclusively on lowercased articles and highlights '
                       'from the training subset. One shared 4,096-token vocabulary is used for both '
                       'sides, including PAD=0, UNK=1, BOS=2 and EOS=3. SentencePiece applies NFKC '
                       'normalization. Source articles are truncated to 400 subwords. Summaries have '
                       'at most 128 tokens including BOS and EOS (126 content subwords). Batches '
                       'are padded to their longest sequence. Original reference highlight newlines '
                       'are retained separately for scoring.'),
             heading('2. Architecture'),
             table([['Setting', 'Value'], ['Encoder / decoder layers', '2 / 2'],
                    ['Model / feed-forward width', '64 / 256'], ['Attention heads', '4 (16 dimensions each)'],
                    ['Dropout / normalization', '0.1 / post-residual LayerNorm'],
                    ['Trainable parameters', f"{training['parameters']:,}"],
                    ['Position / output layer', 'Sinusoidal encoding / linear vocabulary logits']], [210, 297]),
             Spacer(1, 10),
             paragraph('Query, key and value projections are reshaped into heads. Attention is '
                       'softmax(QK^T / sqrt(16) + mask)V, followed by concatenation and a linear '
                       'projection. Feed-forward layers use Linear -> ReLU -> Dropout -> Linear. '
                       'Each sublayer uses LayerNorm(x + Dropout(sublayer(x))). The source padding '
                       'mask is used in the encoder and cross-attention. Decoder self-attention '
                       'combines a triangular causal mask and target-key padding mask. All '
                       'components use elementary tensor operations and basic neural layers.'),
             PageBreak(), heading('3. Training and checkpoint selection'),
             paragraph('Teacher forcing uses gold summaries shifted right as decoder inputs and '
                       'shifted left as targets. Cross-entropy ignores PAD; epoch losses are weighted '
                       'by the number of non-padding target tokens. Adam uses betas=(0.9, 0.98), '
                       'epsilon=1e-9, gradient clipping at 1.0, batch size 16 and seed 42. The learning '
                       f"rate is {training.get('lr_scale',1.0)} * 64^(-0.5) * "
                       'min(step^(-0.5), step * 200^(-1.5)). Early stopping '
                       'has patience 2 after epoch-end validation; the best checkpoint is restored.'),
             paragraph(f"The run completed {len(history)} epochs. Best epoch: {training['best_epoch']}; "
                       f"validation loss: {training['best_validation_loss']:.4f} nats/token. "
                       f"Training time: {training['training_seconds']/60:.1f} minutes on CPU with "
                       f"{training['threads']} PyTorch threads. PyTorch {training['torch_version']}. "
                       'This width and depth are intentionally smaller than the suggested example '
                       'configuration because no GPU is available.'),
             Image(str(RESULTS / 'loss_curves.png'), width=507, height=241),
             paragraph('Selected epochs are tabulated below; all epochs and learning rates '
                       'are available in results/history.csv.', 'SmallBody'),
             table([['Epoch', 'Training loss', 'Validation loss']] +
                   [[row['epoch'], f"{float(row['train_loss']):.4f}",
                     f"{float(row['validation_loss']):.4f}"] for row in shown_history], [110, 198, 199]),
             PageBreak(), heading('4. Full-test quantitative evaluation'),
             paragraph('Greedy decoding begins with BOS, feeds predicted tokens back into the '
                       'decoder, and stops at EOS or 128 generated tokens. PAD and BOS are blocked '
                       'as output tokens. There is no reference input at generation time. Each '
                       'article is encoded once; projected decoder keys and values are cached. '
                       'Caching changes computation cost, not the greedy output (verified by tests).'),
             paragraph('Scores below are macro averages of per-example F1 multiplied by 100 on '
                       'all 11,490 test pairs using rouge_score with use_stemmer=True. ROUGE-L '
                       'uses one sequence; ROUGE-Lsum uses sentence-newline boundaries. Generated '
                       'text and Lead-3 use an offline punctuation-based sentence heuristic; '
                       'reference highlights preserve their existing newlines. This heuristic can '
                       'split abbreviations incorrectly and is a limitation.'),
             table([['System', 'R-1', 'R-2', 'R-L', 'R-Lsum']] +
                   [[name] + [f"{metrics['scores'][key][metric]:.2f}" for metric in
                    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']]
                    for key, name in [('transformer', 'Transformer'), ('lead3', 'Lead-3 (measured)')]] +
                   [['Lead-3 (assignment)', '40.06', '17.48', '25.02', '36.36']], [163, 86, 86, 86, 86]),
             Spacer(1, 12),
             paragraph('Lead-3 copies the first three sentences of each full original article. It '
                       'has access to the full article, whereas the model sees only its first 400 '
                       'subwords. Its measured scores may differ from the assignment reference '
                       'because sentence segmentation is heuristic. Both systems are scored with '
                       'the identical local metric configuration.'),
             paragraph(f"Transformer minus measured Lead-3 ROUGE-1: "
                       f"{metrics['scores']['transformer']['rouge1']-metrics['scores']['lead3']['rouge1']:.2f} "
                       'points. Learning fluent, factual generation from only 2,048 examples is '
                       'much harder than copying informative news leads. The 4,096-piece vocabulary '
                       'also splits rare names and numbers into many fragments, and source '
                       'truncation can remove important details.'),
             paragraph(f'There are {unique_predictions:,} distinct decoded predictions across '
                       'the 11,490 inputs. Distinct outputs do not imply good summaries: repetition, '
                       'omitted facts and unsupported statements must still be assessed. The '
                       'original run collapsed to one output. The revised run reduces the '
                       'learning-rate scale from 1.0 to 0.1 following validation-only diagnostics. '
                       'The model architecture, training subset, tokenizer and seed are unchanged. '
                       'The checkpoint is selected by validation loss, before final test scoring. '
                       'The original test set has already been evaluated, so this is a reused '
                       'benchmark, not a new blind held-out experiment. No test metric selects '
                       'the revised hyperparameters or checkpoint.'),
             paragraph(f"EOS-terminated outputs: {metrics['eos_terminated']:,}/11,490. "
                       f"Outputs with any repeated whitespace trigram: "
                       f"{metrics['repeated_trigram_examples']:,}/11,490. "
                       f"Generation plus scoring took {metrics['evaluation_seconds']/60:.1f} minutes. "
                       'Repetition is measured on unedited decoded outputs; it is not corrected '
                       'with sampling or repetition penalties.'),
             heading('Interpretation and limits'),
             paragraph('ROUGE measures reference overlap rather than factual consistency. Low '
                       'cross-entropy alone cannot establish good summarization because teacher '
                       'forcing supplies correct preceding words during validation. Greedy generation '
                       'must recover from its own errors. This is one seed on a small, ordered '
                       'subset, with a 1,024-example validation subset; no confidence intervals or '
                       'claim of competitive summarization quality are made. More representative '
                       'training data, a wider/deeper model, longer training, and additional validation '
                       'would be the first improvements. Test scores must not guide those choices.'),
             PageBreak(), heading('5. Attention recorded during generation'),
             paragraph('The heatmap uses cross-attention from the last decoder layer for test '
                       'example 1. Four heads are averaged. Each row was recorded while generating '
                       'its labelled predicted token; no gold target was fed to the model. The source '
                       f"window contains 50 consecutive subwords starting at index {attention['start']}, "
                       'chosen to maximize summed attention over all generated steps. For legibility '
                       f"the figure shows the first {attention['shown']} generated subwords. "
                       'Underscores mark SentencePiece word boundaries.'),
             Image(str(RESULTS / 'attention_heatmap.png'), width=507, height=304.2),
             paragraph(f"The selected window contains {100*attention['mass']:.1f}% of attention mass "
                       'per generated row on average (over all generated rows). Probabilities are '
                       'not renormalized after cropping. The complete attention matrix is supplied '
                       'in results/attention.npz. A large weight indicates model focus, not proof '
                       'that a source fact caused a correct prediction. '
                       f"The largest individual weight is {attention['max_weight']:.6f}, "
                       'compared with uniform 1/400 = 0.002500. A uniform 50-token window '
                       'would hold 12.5% of the mass. Concentration alone does not establish '
                       'a correct semantic alignment. '
                       'Repeated rows alongside '
                       'repeated output can indicate a decoding loop; diffuse attention can indicate '
                       'weak alignment. Neither pattern alone proves an error mechanism.'),
             heading('6. Qualitative analysis protocol'),
             paragraph('The next five pages show the first five test examples in fixed repository '
                       'order. This selection was fixed before evaluating outputs and avoids '
                       'cherry-picking. Article excerpts are explicitly labelled; references and '
                       'generated summaries are unedited and complete. A complete per-test '
                       'prediction file is provided separately without articles. The diagnostics '
                       'below each example describe repetition and overlap; they do not certify facts.')]
    for example in examples:
        story += [PageBreak(), heading(f"7.{example['index']+1}. Test example {example['index']+1}"),
                  paragraph('Dataset ID: ' + example['id'], 'SmallBody'),
                  paragraph('Article excerpt (first 1,100 characters)', 'Heading2'),
                  paragraph(example['article_excerpt'], 'SmallBody'),
                  paragraph('Reference highlights (original newlines)', 'Heading2'),
                  paragraph(example['reference'], 'SmallBody'),
                  paragraph('Generated summary (unedited greedy output)', 'Heading2'),
                  paragraph(example['generated'] or '[Empty output: EOS immediately]', 'SmallBody'),
                  paragraph('Observed error indicators', 'Heading2'),
                  paragraph(example_analysis(example), 'SmallBody')]
    original = json.loads((RESULTS / 'audit/diagnostics.json').read_text())
    revised = json.loads((RESULTS / 'audit_revised/diagnostics.json').read_text())
    story += [PageBreak(), heading('8. Collapse investigation and correction'),
              paragraph('The original checkpoint was audited on validation indices '
                        '0, 1, 2, 100, 500 and 1000. All six source token sequences were different. '
                        'Original encoder layer 1 retained differences (RMS 0.86-0.87), but layer 2 '
                        'reduced them to about 0.0013. Fresh random weights produced encoder '
                        'differences around 1.36-1.37. This is evidence of learned representation '
                        'collapse rather than repeated tokenized input.'),
              paragraph('Cached and uncached greedy IDs matched exactly for all six real '
                        'validation articles, including per-article versus batch runs and a '
                        'shortened validation source to exercise padding. Hooks checked both '
                        'decoder layers at every generation step: encoder K/V values and source '
                        'masks were correct; blocked positions had zero attention. Cached versus '
                        f"full-prefix logit error was at most {original['max_cached_vs_full_logit_error']:.2g}. "
                        'The complete original validation loss recomputed to exactly 6.0200552, '
                        'matching the checkpoint. No decoder routing or cache defect was found.'),
              paragraph('The original schedule peaked at 0.0088388, which is aggressive for this '
                        'small post-normalization model. A controlled retraining uses a 0.1 scale '
                        '(peak 0.00088388), a maximum of 12 epochs and the same patience of 2. '
                        'No architecture or decoding changes are introduced to force diverse '
                        'outputs. The full audit is repeated on the revised validation checkpoint.'),
              table([['Validation diagnostic', 'Original', 'Revised'],
                     ['Distinct input sequences / 6', original['unique_sources'], revised['unique_sources']],
                     ['Distinct generated sequences / 6', original['unique_validation_predictions'], revised['unique_validation_predictions']],
                     ['Encoder RMS: article 1 vs 2', f"{original['trained_memory_rms_vs_first'][0]:.6f}", f"{revised['trained_memory_rms_vs_first'][0]:.6f}"],
                     ['Cached = uncached', original['cached_equals_uncached'], revised['cached_equals_uncached']],
                     ['Recomputed validation loss', f"{original['validation_loss_recomputed']:.6f}", f"{revised['validation_loss_recomputed']:.6f}"]], [235,136,136]),
              Spacer(1, 10),
              paragraph('The original data/tokenizer reuse was valid, but the scripts failed to '
                        'check it. The corrected scripts validate counts, unique/disjoint IDs, '
                        'portable content hashes, BPE vocabulary, special token IDs and tokenizer '
                        'fingerprints. Fresh streams from pinned revision '
                        '96df5e686bee6baa90b8bee7c28b81fa3fa6223d matched all three saved subsets. '
                        'Training-only tokenizer reconstruction reproduced every piece and score.'),
              PageBreak(), heading('9. Stability comparison'),
              Image(str(RESULTS / 'stability_comparison.png'), width=430, height=204.25),
              paragraph('The comparison above uses validation only. The original run can have '
                        'a lower teacher-forced loss while ignoring its source: the correct '
                        'preceding target words still help next-token prediction. Restoring '
                        'source sensitivity does not itself guarantee a lower loss, fluent '
                        'generation or accurate facts. The revised checkpoint is chosen by '
                        'the lowest validation loss within the revised, fixed training run. '
                        'The unchanged small data sample remains a major limitation.'),
              heading('10. Verification and reproduction'),
              paragraph('Unit tests verify future-token isolation, source and target padding '
                        'masking, attention normalization, ignored padding loss, summary shifting, '
                        'sinusoidal positions, optimization on a fixed mini-batch, sentence '
                        'boundaries, and equality of cached and full-prefix greedy predictions. '
                        'See test_results.txt for the recorded run and README.md for commands.'),
              paragraph('Observed host: Python 3.14.8 on Windows 11; PyTorch runtime '
                        '2.14.1+cpu (distribution metadata 2.14.1), NumPy 2.5.4 in the local '
                        'dependency overlay, datasets 5.1.0, SentencePiece 0.2.2, rouge-score '
                        '0.1.2, matplotlib 3.11.2, reportlab 5.1.0, nltk 3.10.4 and protobuf '
                        '7.36.2. The original training run used base NumPy 2.5.3; this revised '
                        'run uses 2.5.4. PyTorch is unchanged. A separate virtual environment '
                        'with these pinned direct dependencies and system-site-packages passed '
                        'pip check, all 14 tests, dataset verification and the offline downloader '
                        'reuse path. A fully isolated clean-machine installation is not claimed.'),
              paragraph('The archive includes source scripts, this report, measured metrics, '
                        'training history, figures, five qualitative examples, test predictions and '
                        'generation-time attention. It excludes dataset downloads, model checkpoints, '
                        'the tokenizer training corpus and installed dependencies. Tokenizer and '
                        'model artifacts are regenerated in a separate work directory.'),
              heading('References'),
              paragraph('Assignment: NLP Group Assignment 2, DSL 504, Question 3, pages 8-11.', 'SmallBody'),
              paragraph('Vaswani et al. (2017). Attention Is All You Need. '
                        'https://arxiv.org/abs/1706.03762', 'SmallBody'),
              paragraph('CNN / Daily Mail 3.0.0, canonical Hugging Face repository: '
                        'https://huggingface.co/datasets/abisee/cnn_dailymail', 'SmallBody'),
              paragraph('SentencePiece: https://github.com/google/sentencepiece', 'SmallBody'),
              paragraph('ROUGE implementation: https://github.com/google-research/google-research/tree/master/rouge', 'SmallBody'),
              paragraph('PyTorch softmax documentation: '
                        'https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html', 'SmallBody')]
    document = SimpleDocTemplate(str(ROOT / 'Final_Report.pdf'), pagesize=(595.28, 841.89),
                                 rightMargin=44, leftMargin=44, topMargin=40, bottomMargin=44)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print('Created Final_Report.pdf and figures.')


if __name__ == '__main__':
    main()
