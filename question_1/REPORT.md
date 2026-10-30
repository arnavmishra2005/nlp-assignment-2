# NLP Group Assignment 2: Does attention help English to Hindi translation?

Short technical report. Full code, plots and analysis are in `NLP_Group_Assignment_2.ipynb`.

## What was built

| File | Purpose |
|---|---|
| `data.py` | Part 1: load IIT Bombay pairs, clean, tokenize, add `<SOS>`/`<EOS>`, vocabularies, ids, padding |
| `models.py` | Encoder, Attention, Decoder and `Translator(use_attention=False/True)` (Model A and Model B) |
| `train.py` | Training with teacher forcing, per-epoch loss and accuracy, best epoch kept by validation loss |
| `analyze.py` | BLEU, length groups, error types, attention heatmaps, plots |
| `NLP_Group_Assignment_2.ipynb` | Parts 1 to 9 with outputs and written analysis |
| `results/` | training curves, heatmaps, error case heatmaps, length plot, all test translations (`test_translations.csv`), saved models |
| `results_run1_30k/` | the earlier, smaller run (30,000 pairs, 6 epochs), kept for comparison |

Run: `python train.py baseline 8 6` and `python train.py attention 8 6` (model, epochs, CPU threads; run in parallel), then open the notebook.

## Setup

- **Data:** 50,000 training pairs (random sample of 200,000 raw IIT Bombay pairs, at most 25 tokens per side), 430 dev pairs and 1,723 test pairs (up to 30 tokens). Vocabularies: 10,000 words each (English and Hindi), everything else is `<UNK>` (7.2% of English and 9.8% of Hindi training words). Longest sentence: 27 tokens with `<SOS>` and `<EOS>`.
- **Model A (baseline):** embedding (128) then LSTM encoder (256), final state starts an LSTM decoder, then a linear layer with softmax. 5.9M parameters.
- **Model B (attention):** the same, but the encoder keeps all hidden states h1..hn. At every step the score `v . tanh(W_s s + W_h h_i)` is turned into weights alpha by a softmax and the weighted sum of the h_i is the context vector fed into the decoder (6.3M parameters).
- **Same conditions:** Adam, learning rate 0.002, batch 128, dropout 0.3, 8 epochs, same seed. Evaluation uses greedy decoding with `<UNK>` not allowed.

## Results (test set, 1,723 sentences)

| | Baseline | Attention |
|---|---|---|
| Validation loss / word accuracy (best epoch) | 4.760 / 0.289 | 4.654 / 0.302 |
| BLEU (0 to 100) | 0.60 | 0.92 |
| Word overlap F1 with the reference | 0.187 | 0.205 |

By English sentence length (BLEU, baseline vs attention): 1-5 tokens (only 10 sentences) 1.97 vs 2.50; 6-10: 1.25 vs 1.14; 11-15: 0.56 vs 1.09; 16+: 0.49 vs 0.82. Baseline word accuracy falls steadily with length (0.44, 0.34, 0.30, 0.27).

**Effect of more data and epochs** (first run: 30,000 pairs, 6 epochs, vocabulary 8,000, batch 64):

| | Baseline | Attention | Attention gain in word overlap (± std. error) |
|---|---|---|---|
| Run 1: BLEU / overlap F1 | 0.41 / 0.173 | 0.56 / 0.183 | 0.0095 ± 0.0021 |
| Run 2: BLEU / overlap F1 | 0.60 / 0.187 | 0.92 / 0.205 | 0.0178 ± 0.0024 |

## Findings

1. **Attention helps, and more so with more data.** Attention is better on 926 test sentences, worse on 718 and equal on 79 (mean word-overlap gain 0.0178 ± 0.0024). It is better for every length group by word overlap, and clearly better on BLEU and accuracy from 11 tokens up.
2. **The baseline gets worse with length**, which fits the idea of squeezing a whole sentence into one vector. There is no clear sign that early words are lost (the end of a Hindi sentence is mostly है and ।).
3. **Heatmaps:** the first words align well (but gives लेकिन, she gives वह, "he said" gives "उन्होंने कहा", but gives बल्कि), then attention spreads out (in 86% of steps no English word gets more than 0.5). Sharp attention on an unknown word (asaram, diwali, inappropriately) leads to a generic wrong word.
4. **Attention is only a partial explanation.** It shows where a translation goes wrong (looking at an unknown word, spreading out on long sentences), but sharp attention does not mean a good translation (correlation 0.20), and a clean alignment can still drop a word.
5. **Errors:** attention reduces missing words (43.0% to 38.2% of translations) but increases repeated words (37.1% to 46.3%) and extra words (29.5% to 34.3%).

## Limitations

Only 50,000 pairs, 8 epochs and one run per model; word-level vocabulary (about 10% of Hindi training words are unknown); greedy decoding; `<UNK>` banned at decoding (without this, 57% and 67% of the output words are `<UNK>`); BLEU is below 1 for both models, so these are translations of common phrases and not good translations; small dev set; only 10 test sentences in the 1-5 group; error types are rule based guesses. Subword tokens, much more data and epochs, and beam search would raise the absolute quality.
