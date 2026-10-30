# Question 3: Abstractive Text Summarization

This submission answers **only Question 3**. It implements a full Transformer
encoder-decoder using elementary PyTorch tensor operations and layers. All model
weights start randomly. No pretrained summarization model is used.

## Setup

The tested host is Windows 11, Python 3.14.8, CPU PyTorch runtime
`2.14.1+cpu` (distribution metadata `2.14.1`). Exact observed direct dependency
versions are pinned in `requirements.txt`; module paths and versions are recorded
in `results/audit/environment.json`. Other Python versions were not tested.
Open a terminal in this directory:

```powershell
python -m venv --system-site-packages .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

This is the setup actually tested: the virtual environment reuses the host's
existing CPU PyTorch and other compatible packages, and installs missing pinned
dependencies locally. `pip check` passed. A completely isolated fresh-machine
installation was not tested. On a machine without these dependencies, create a
normal venv instead and install the requirements; availability of the exact
recorded releases must be checked with your package index.

## Reproduce the experiment

Internet access is needed to download the public dataset. The download uses
`abisee/cnn_dailymail`, the canonical repository for CNN/Daily Mail, version 3.0.0.
Keep the working directory outside this submission folder:

```powershell
python download_data.py --work ../work --train-size 2048 --validation-size 1024
python -m unittest discover -s . -p test_solution.py -v
python train.py --work ../work --output results --epochs 12 --patience 2 --batch-size 16 --threads 4 --lr-scale 0.1
python verify_inputs.py --work ../work
python audit_collapse.py --work ../work --output results/audit_revised --lr-scale 0.1
python evaluate.py --work ../work --output results --batch-size 32 --threads 4
python make_report.py
```

`verify_inputs.py` runs after training has created the tokenizer. Training itself
verifies the training/validation files and tokenizer automatically. The
downloader verifies files on reuse, and pins the dataset revision to
`96df5e686bee6baa90b8bee7c28b81fa3fa6223d`. For an independent online content check:

```powershell
python download_data.py --work ../work --verify-source
```

This online check was completed for all three existing subsets. It verifies data
provenance and does not compute test scores. `input_manifest.json` fingerprints
the declared subsets and tokenizer; changed data/order/vocabulary is rejected.
The manifest is intentionally fixed to this experiment's subset sizes.

Run evaluation only after training/model selection is complete. It requires all
11,490 test examples and restores the checkpoint with the lowest validation loss.
It generates without reference summaries, computes stemmed ROUGE-1/2/L/Lsum F1,
and scores a Lead-3 baseline using the same scorer. Predictions are lowercased;
references are the original highlights with their sentence newlines preserved.
No test score is used to choose training or generation settings.

The report and measured result files describe the completed run. CPU execution
can take tens of minutes. A CUDA GPU is selected automatically when available.
Seed 42 is fixed; exact numerical reproducibility across hardware/PyTorch versions
is not guaranteed.

## Files

- `data.py`: lowercasing, training-only SentencePiece BPE, truncation, padding and shifting.
- `model.py`: attention, sinusoidal positions, feed-forward networks, encoder,
  decoder and cached greedy generation.
- `train.py`: teacher forcing, Adam warmup/decay, loss history and early stopping.
- `evaluate.py`: full-test generation, four ROUGE metrics, Lead-3, examples and attention.
- `make_report.py`: measured plots, attention heatmap and final PDF report.
- `test_solution.py`: correctness tests for masks, losses, shifting and inference.
- `audit_collapse.py`: six real validation articles, cache/logit comparison,
  memory/mask routing, representation/gradient diagnostics, loss reproduction
  and training-only tokenizer reconstruction; it does not update model weights.
- `verify_inputs.py`, `input_manifest.json`: semantic data and tokenizer checks.
- `results/`: measured history, settings, metrics, figures, examples, attention
  arrays and generated test predictions (IDs and model output only).
- `Final_Report.pdf`: methodology, results, plots, limitations and qualitative examples.
- `test_results.txt`: actual test output.

## Review correction

The original model had learned encoder representation collapse. Six validation
inputs were different; the last encoder layer reduced their RMS differences to
about 0.0013. Cached and uncached generation matched exactly, correct source K/V
and masks reached both decoder layers, and the saved validation loss reproduced
exactly. The original tokenizer was rebuilt from training-only text and every
piece and score matched. The original data also matched fresh pinned downloads.

The revised experiment keeps the architecture, seed, tokenizer and subsets and
reduces the schedule scale from 1.0 to 0.1 (peak LR 0.00088388). It trains for at
most 12 epochs, stopping after two non-improving validation epochs and restoring
the best validation checkpoint. Original audit evidence is in `results/audit/`;
revised evidence is in `results/audit_revised/`. No revised setting is selected
using test ROUGE. The test benchmark has been used before, so revised scores are
not a new blind held-out experiment. Distinct outputs are not evidence of factual
or fluent summaries; see the updated report for remaining failures.

The supplied correctness tests, manifest checks and offline downloader-reuse path
were executed in the pinned virtual environment. Full revised training is run in
the recorded host-plus-local-dependencies environment. Exact cross-platform
numerical replay is not claimed. The PDF is regenerated from the measured result
files; its original-collapse discussion also reads the bundled original audit.

## Experiment choices

The CPU run uses the first 2,048 training and 1,024 validation pairs, not random
samples. The entire 11,490-pair test split is retained. SentencePiece BPE has 4,096
shared entries and is trained only on lowercased training text. Source limit is
400 tokens; target limit is 128 including BOS/EOS. Generation allows 128 new tokens.
The model has two layers per stack, width 64, four heads, FFN width 256 and dropout
0.1. See `results/training.json` for the measured parameter count and selected epoch.
This intentionally small experiment is expected to produce weak summaries.

The heatmap averages the last decoder layer's four cross-attention heads recorded
during generation. It shows a 50-source-subword window with maximum total attention
and the first 24 output subwords; the full matrix is in `results/attention.npz`.
Sentence splitting uses a documented offline punctuation heuristic, so measured
Lead-3 can differ from the assignment's reference implementation.

## Submission contents

Submit the supplied ZIP. Do **not** add `../work`, datasets, checkpoints,
dependencies, virtual environments or Python caches. The tokenizer model is
regenerated from training data and is stored in `../work`, with the checkpoint.
Five short attributed qualitative excerpts are included for the required analysis;
the dataset itself is not distributed.
