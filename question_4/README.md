# Plot-Twist Generator

Vanilla RNN vs LSTM language models trained on TinyStories and used to generate story continuations.

**Course:** DSL 504 NLP Group Assignment 2, Question 4

## 1. Overview

| Model | Architecture |
|---|---|
| Vanilla RNN | Embedding -> `nn.RNN` -> Linear |
| LSTM | Embedding -> `nn.LSTM` -> Linear |

Both models share the vocabulary, data splits and training setup. They are compared on loss, perplexity, lexical diversity, repetition and generated story continuations. Decoding strategies: greedy, temperature sampling, top-k sampling.

## 2. Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS: `source .venv/bin/activate`

Run every command from the `question_4` directory as `python -m src.<module>` (not from the overall assignment root).

GPU: install a CUDA build of PyTorch from https://pytorch.org/get-started/locally. The code uses CUDA when available and falls back to CPU. For Google Colab, see `COLAB_STEPS.md`.

## 3. Data preparation

Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). The raw dataset and processed token-id splits are not included in the submission archive. Preprocessing downloads the selected stream from Hugging Face; an internet connection is required for that step.

```cmd
python -m src.preprocess
```

| Setting | Value |
|---|---|
| Source | First 22,000 stories of the TinyStories train stream |
| Stories (train / val / test) | 20,000 / 1,000 / 1,000 |
| Split level | Story level, disjoint, shuffled with seed 42 |
| Tokenisation | Word-level regex, lowercased, curly quotes normalised |
| Vocabulary | Train split only, min frequency 5, max 10,000 (6,695 words after filtering) |
| Special tokens | `<pad>=0 <unk>=1 <bos>=2 <eos>=3` |
| Max story length | 200 tokens (longer stories truncated, no `<eos>` appended) |
| Seed | 42 |

Each story is its own sequence (`<bos> ... <eos>`). Stories are never concatenated, so there are no fake transitions between unrelated stories and no leakage across splits.

Batches are padded with `<pad>`. Loss uses `CrossEntropyLoss(ignore_index=pad_id)`, so padding is never counted.

| Path | Content |
|---|---|
| `data/processed/{train,val,test}.pt` | Generated locally by preprocessing; intentionally excluded from the submission ZIP |
| `artifacts/vocab.json` | Vocabulary |
| `artifacts/data_stats.json` | Sizes, vocab size, unk rate, truncation |

| Split | Stories | Tokens | Avg length | `<unk>` rate | Truncated stories |
|---|---|---|---|---|---|
| Train | 20,000 | 3,483,783 | 174.2 | 0.22% | 32.7% |
| Val | 1,000 | 173,213 | 173.2 | 0.34% | 30.7% |
| Test | 1,000 | 174,787 | 174.8 | 0.34% | 32.2% |

Known limitation: text cleaning does not remove some garbled characters in the raw data, so a few tokens such as `€` and `™` are in the vocabulary and appear in some generations.

## 4. Training

```cmd
python -m src.train --model rnn
python -m src.train --model lstm
python -m src.plot_curves
```

| Setting | Value (both models) |
|---|---|
| Embedding / hidden dim | 256 / 512 |
| Layers | 1 |
| Dropout | 0.2 |
| Optimizer | Adam, lr 1e-3 |
| Gradient clipping | 1.0 |
| Batch size | 64 |
| Max epochs / patience | 10 / 3 (early stopping on val loss) |
| Loss | Cross-entropy, padding ignored, averaged per real token |
| Seed | 42 |

### Results

| Metric | RNN | LSTM |
|---|---|---|
| Parameters | 5,542,695 | 6,725,415 |
| Training time | 344 s | 473 s |
| Epochs run | 10 | 10 |
| Best val loss | 2.8372 | 2.6092 |
| Val perplexity | 17.07 | 13.59 |
| Test loss | 2.8261 | 2.5955 |
| Test perplexity | 16.88 | 13.40 |

Training times depend on the hardware used. Early stopping never triggered. Validation loss was still falling at epoch 10 for both models, so both are undertrained under the 10-epoch cap.

| Path | Content |
|---|---|
| `checkpoints/{rnn,lstm}.pt` | Best checkpoint by val loss (included in the submitted archive) |
| `artifacts/{rnn,lstm}_history.json` | Losses, perplexity, parameters, training time, test perplexity |
| `artifacts/loss_curves.png` | Train and validation loss curves |

## 5. Generation

Example commands (the `--model` option is required):

```cmd
python -m src.generate --model rnn --strategy greedy
python -m src.generate --model rnn --strategy temperature --temperature 0.5
python -m src.generate --model rnn --strategy temperature --temperature 0.8
python -m src.generate --model rnn --strategy temperature --temperature 1.2
python -m src.generate --model rnn --strategy top_k --top_k 10
```

Repeat with `--model lstm`.

| Strategy | Description |
|---|---|
| Greedy | Highest-probability token at every step |
| Temperature | Logits divided by T before sampling, T in 0.5, 0.8, 1.2 |
| Top-k | Sample from the k most probable tokens, k = 10 (default), T = 1.0 |

| Generation parameter | Value |
|---|---|
| Max new tokens | 80 (`--max_new_tokens`) |
| Stop rule | `<eos>` or max new tokens |
| Blocked tokens | `<pad>`, `<bos>` |
| Seed | 42 (`--seed`) |
| Prompts | 7, defined in `src/generate.py` (mystery, discovery, unusual character, problem to solve, and others) |

Output: one JSON file per run in `results/`, named `{model}_{strategy}_t{temperature}_k{top_k}.json`. Each record stores model, strategy, temperature, top_k, seed, prompt and output. For greedy runs the temperature value in the file name is ignored by the decoder.

## 6. Plot-Twist Challenge

```cmd
python -m src.challenge
```

Seven twist openings (exam paper with tomorrow's answers, robot with a message from its future self, detective solving a murder before it happens, village forgetting the previous day, letter from 100 years in the future, new kid who knows everyone's secrets, moon moving closer to the village), defined in `src/challenge.py`.

Each opening is run with both models and three strategies: greedy, temperature 0.8, top-k (k = 10). Two samples are generated per setting, so `results/plot_twist_challenge.json` has 84 records. Greedy decoding is deterministic, so its two samples are identical (70 unique outputs). Outputs are kept unedited, including failures. Max new tokens is 80 and the seed is set once at the start (42).

## 7. Evaluation

```cmd
python -m src.metrics
```

`src.metrics` requires `data/processed/test.pt`, so run preprocessing first. The included `artifacts/metrics.json` preserves the recorded metrics from the submitted checkpoints; rerunning evaluation may require downloading the dataset and rebuilding the processed splits.

Output: `artifacts/metrics.json`.

| Metric | Definition |
|---|---|
| Val / test perplexity | `exp(mean per-token cross-entropy)` |
| Distinct-1 / Distinct-2 | Unique unigrams (bigrams) divided by total unigrams (bigrams) |
| Repetition rate | `1 - unique 3-grams / total 3-grams` per output, averaged |
| Average generated length | Mean words per output |
| Premature / overlong outputs | Outputs under 20 words / outputs of 180 words or more |

Notes on the generation metrics:
- They are computed over the 7 prompts in `src/generate.py` and 5 settings per prompt (greedy, T = 0.5, 0.8, 1.2, top-k), so 35 outputs per model.
- The prompt is included in the text that is measured.
- The metrics script seeds PyTorch sampling with 42. Exact results can still vary across software/hardware versions. Perplexity does not depend on sampling.
- With an 80-token cap, outputs of 180 words or more cannot occur, so the overlong count is always 0.

The report discusses strategy trade-offs qualitatively. The supplied pooled metric table combines strategies; it is not a per-strategy quantitative comparison.

## 8. Human evaluation

Run:

```bash
python -m src.human_eval
```

This creates three files:

- `results/human_eval_blinded.csv`: 30 samples with the prompt and continuation, and blank rating fields.
- `results/human_eval_3_rater_template.csv`: 90 rows, one row per sample per each of three rater IDs.
- `results/human_eval_key.csv`: the separate mapping from sample IDs to model and decoding strategy. **Do not share this key with raters until they finish.**

The rating cells are intentionally blank: no genuine human ratings were supplied with the project. The human-evaluation requirement is not complete until real evaluators score the samples. See `results/HUMAN_EVALUATION_INSTRUCTIONS.md` for the scoring workflow. Do not report blank cells as results or describe them as completed human evaluation.

## 9. Limitations

- Small word-level models trained on a 20k-story subset, undertrained under the 10-epoch cap, single seed.
- Stories truncated at 200 tokens, so long-range structure is limited.
- Observed failures: repetition loops (especially greedy), drifting characters, lost prompt context, broken characters and `<unk>` tokens, nonsense at high temperature.
- Higher diversity and lower perplexity do not guarantee better storytelling.
- Small evaluation: 7 prompts, few samples per setting.

## 10. Project structure

```text
plot-twist-generator/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── plot_curves.py
│   ├── generate.py
│   ├── challenge.py
│   ├── metrics.py
│   └── human_eval.py
├── artifacts/
│   ├── vocab.json
│   ├── data_stats.json
│   ├── rnn_history.json
│   ├── lstm_history.json
│   ├── loss_curves.png
│   └── metrics.json
├── checkpoints/
│   ├── rnn.pt
│   └── lstm.pt
├── results/
│   ├── {model}_{strategy}_t{T}_k{K}.json
│   ├── plot_twist_challenge.json
│   ├── human_eval_blinded.csv
│   ├── human_eval_3_rater_template.csv
│   ├── human_eval_key.csv
│   └── HUMAN_EVALUATION_INSTRUCTIONS.md
├── data/README.md
├── Plot_Twist_Generator_Final_Report.docx
├── Plot_Twist_Generator_Final_Report.pdf
├── requirements.txt
├── COLAB_STEPS.md
└── README.md
```

`data/processed/` is created by `preprocess` and is not submitted.

## 11. Reproducibility

1. Install dependencies.
2. `python -m src.preprocess`
3. `python -m src.train --model rnn` and `python -m src.train --model lstm`
4. `python -m src.plot_curves`
5. `python -m src.generate` runs (see section 5) for each model and setting
6. `python -m src.challenge`
7. `python -m src.metrics`
8. `python -m src.human_eval`

Seed 42 is used for data splitting, training and generation. Exact numbers can vary with PyTorch version, hardware and CUDA. The project-trained checkpoints and vocabulary are included, so generation, the challenge and rating-form creation can run without retraining. Metrics reevaluation additionally requires preprocessing to recreate the held-out test data.


## Human evaluation status (important)

The package contains a blinded form for 30 samples, a three-rater template, and a separate sample-to-model key. **The rating cells are intentionally blank because no genuine human ratings have been supplied.** The assignment's human-evaluation component is not complete until real evaluators independently score the samples. Do not report blank cells as results and do not share `results/human_eval_key.csv` with raters before they finish.

Create or refresh the forms with:

```bash
python -m src.human_eval
```

See `results/HUMAN_EVALUATION_INSTRUCTIONS.md` for the scoring rubric and workflow.

## Files included in the submission

- `src/`: preprocessing, model definitions, training, generation, challenge, metrics, and human-evaluation form generation.
- `checkpoints/`: trained RNN and LSTM state dictionaries.
- `artifacts/`: vocabulary, data statistics, training histories, metrics, and loss curves.
- `results/`: generated continuations, challenge outputs, and blinded human-rating materials.
- `Plot_Twist_Generator_Final_Report.docx` and `.pdf`: final technical report.
- `COLAB_STEPS.md`, `requirements.txt`, and this README: setup and reproduction guidance.

## Continuation-length measurements

Supplemental continuation-only lengths and generation-cap counts from the saved general outputs are in `artifacts/continuation_length_audit.json`. These use the saved-output corpus, separately from the sampled original pooled-metrics corpus.
