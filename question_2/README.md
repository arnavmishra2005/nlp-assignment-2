# Question 2 - Amazon review rating prediction

## Submitted work

Compare a pretrained-GloVe, two-layer bidirectional GRU with additive attention against fully fine-tuned RoBERTa-base with masked mean pooling. First-token pooling is the ablation. Both seeds (42 and 123) were completed for all three variants.

| Test metric: mean over two seeds | BiGRU | RoBERTa mean |
|---|---|---|
| Accuracy | 63.79% | 67.13% |
| Macro F1 | 0.6351 | 0.6706 |
| MAE | 0.4277 | 0.3672 |

Evaluation inspection passed for 89 files; CPU finalization passed for 229 files. These archived-phase counts describe the original verified exports, not the number of files after folder organization. The six-page report and executed notebooks are included here.

## Directory guide

| Path | Content |
|---|---|
| `q2/`, `config.json` | Model, preprocessing, training, distributed execution, frozen evaluation, analysis, metrics, provenance and artifact checks |
| `kaggle_start.py`, setup/download/helper scripts | Command-line staged execution and environment management |
| [KAGGLE_RUN_GUIDE.md](KAGGLE_RUN_GUIDE.md) | Phase inputs, settings, CLI commands, saved execution and recovery instructions |
| [notebooks/executed/README.md](notebooks/executed/README.md) | Nine actual successful Kaggle executions; phase/source/config/split receipts verified; local CPU-finalization JSON evidence |
| `notebooks/*.ipynb` | Unexecuted three-cell templates for reproducing phases |
| `results/` | Frozen per-seed predictions/metrics, aggregate tables, confusion matrices, training curves, ablation, qualifying examples and importance scores |
| `training_records/` | Six run configurations, histories, completion records, logs and actual environment records |
| [report/Q2_Technical_Report.pdf](report/Q2_Technical_Report.pdf) | Exactly six pages, including tables/figures, five forward and two reverse comparison cases |
| `report/build_q2_report.py` | Reproducible report builder from the saved measured results; requires reportlab and pypdf |
| `report/Q2_qualitative_examples_annotated.csv` | Separate interpretation copy; inspected raw examples in results are unchanged |
| `tests/`, [VERIFICATION.md](VERIFICATION.md) | Source correctness checks and documented verification scope |

## Dataset and preprocessing

Use the official 200,000/5,000/5,000 train/validation/test splits of SetFit/amazon_reviews_multi_en. SetFit omits titles, so recover them from buruzaemon's original-corpus mirror using exact review-ID/body/language/star agreement, then concatenate title + newline + body. The exact repository IDs/revisions are in `config.json`; no dataset-wide body-only fallback is allowed. Original missing titles remain empty and are counted (33 train, one validation). Five exact train/validation text overlaps are disclosed while retaining the required official splits.

The classic pipeline HTML-unescapes, lowercases, removes tags/URLs, and regex-tokenizes. Its training-only vocabulary uses minimum frequency 2 (32,611 entries) and max length 200. GloVe 6B 100d coverage is 87.77%; OOV initialization uses fixed seed 2026. RoBERTa uses the original concatenated text, RobertaTokenizerFast, max length 256 and dynamic padding to multiples of eight. Review IDs, class counts, data revisions and split fingerprints are checked before execution.

## Training and evaluation protocol

BiGRU uses two layers, 128 hidden units per direction, additive attention width 128, dropout 0.3, Adam lr 0.001, batch 128, epoch cap 8 and patience 2. GloVe is frozen for the first epoch then unfrozen. Packed sequences and attention masks exclude padding.

RoBERTa uses dropout 0.2, AdamW lr 0.00002, decay 0.01, 8% warmup then linear decay, global microbatch 16 and accumulation 2 (effective batch 32), epoch cap 3 and patience 1. Both use cross-entropy, clipping 1.0 and validation Macro F1 selection. Actual precision and hardware are recorded per run.

The retained mean-seed42 run used one GPU; the other three RoBERTa runs used two-T4 DistributedDataParallel. Effective batch and update count were preserved, but rank-specific dropout/reduction changes mean numerical equivalence is not claimed. This limits pooling-only attribution for the seed42 pair; the seed123 pair shares topology.

All six selected runs were frozen before final test access. The baseline and mean model were scored once per seed on 5,000 test examples. The first-token ablation uses validation results, not additional test-driven model selection. Aggregates use sample SD (ddof=1), not confidence intervals. Adjacent errors mean rating distance 1, distant errors >=2; rates and shares use separately labelled denominators. Gradient x input is local sensitivity, not causal proof.

## Reproduce or inspect

Use a separate environment from this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The requirements preserve the installed CUDA-enabled PyTorch stack in Kaggle; setup checks and installs only required compatible non-CUDA dependencies. For a fresh reproduction, use the phase sequence and exact arguments in KAGGLE_RUN_GUIDE.md: prepare -> smoke -> six training phases -> evaluate -> finalize. To create a Kaggle source upload from this folder, run `python tools/package.py --output dist/q2_source.zip`. Follow the guide's explicit source/predecessor input handling. Actual executed notebooks are evidence; source templates launch fresh runs.

To rebuild the report from repository root:

```powershell
python question_2/report/build_q2_report.py
```

The report builder requires `reportlab` and `pypdf`; it checks the six-page limit. No training or test-data access is needed to inspect supplied CSVs, figures, notebooks or the PDF. Do not change settings against the measured test results or portray an unexecuted template as executed evidence.

## Packaging and review

Full datasets, caches, live recovery states and large Q2 checkpoints are excluded from submission. Original selected models and download archives are retained by the coordinator in the excluded working archive. A recipient can reproduce training from the documented pinned sources; optional prediction requires the appropriate retained/trained checkpoint.

The raw selected examples are in `results/qualitative_examples.csv`; the six-page section and separate annotated examples are in `report/`. The original finalization records document archive generation; the combined report contains the technical analysis.

## Sources

- [Specified SetFit dataset](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en)
- [Original-corpus mirror for title recovery](https://huggingface.co/datasets/buruzaemon/amazon_reviews_multi)
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base)

Exact executed configurations, revisions, software versions and selection evidence are retained in the supplied records.
