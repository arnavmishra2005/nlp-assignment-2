# NLP Group Assignment 2

**DSL 504 - Natural Language Processing**  
**Repository:** [arnavmishra2005/nlp-assignment-2](https://github.com/arnavmishra2005/nlp-assignment-2)  

## Team members

| Name | ID | Primary responsibility |
|---|---|---|
| **Rudra Mehul Dudhat** | B24DS506 | Q1: English-Hindi LSTM translation, attention, experiments and report |
| **Arnav Mishra** | 12340330 | Q2: review-rating classification, staged experiments and report; group integration/submission |
| **Vatsal Yadav** | B24DS036 | Q3: from-scratch Transformer summarization, collapse investigation and revised report |
| **Mohak Arya** | 12341420 | Q4: RNN/LSTM plot-twist generation, evaluation and report, jointly with Kanishk |
| **Kanishk Nandeshwar** | B24DS010 | Q4: RNN/LSTM plot-twist generation, evaluation and report, jointly with Mohak |

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for the contribution split and question-level deliverables.

## Combined technical report

**[Open the combined technical report](report/Combined_Technical_Report.pdf)** - 30 pages, with bookmarks for all four questions. [Report page/check record](report/combined_report_check.json).

| Section | Combined PDF pages | Included evidence |
|---|---|---|
| Q1: attention-based translation | 2-6 | Method, curves, metrics, length analysis and six translation/attention cases |
| Q2: Amazon review ratings | 7-12 | Exactly six pages: training, final metrics, ablation, seven examples and token importance |
| Q3: scratch Transformer summarization | 13-23 | Revised experiment, ROUGE/Lead-3, generation attention, five cases, diagnosis and limitations |
| Q4: plot-twist generator | 24-29 | Both models, decoding/challenge, quantitative results, failures and explicit human-evaluation status |

Page 1 contains the group overview; page 30 summarizes the experimental comparison and limitations. Q2 occupies exactly six pages, including figures and tables.

## Repository structure

```text
nlp-assignment-2/
|-- README.md
|-- CONTRIBUTIONS.md
|-- question_1/       source, executed notebook, results, figures and Q1 report
|-- question_2/       source/config, results, training records, notebooks and six-page report
|-- question_3/       revised scratch-Transformer code, tests, results, diagnosis and report
|-- question_4/       source, two required checkpoints, vocabulary, results and rating materials
|-- report/           combined PDF, reproducible builder and page/status checks
`-- _working_archive/ local backups/intermediate runs - NOT pushed or submitted
```

This README is the combined entry point. Each question also has a detailed README because datasets, commands and dependency versions differ. There is deliberately no single shared environment requirement file.

## Quick start - Windows, Linux and macOS

```bash
git clone https://github.com/arnavmishra2005/nlp-assignment-2.git
cd nlp-assignment-2
```

Open the report, saved figures/CSVs and executed notebooks to inspect existing results without training. For running code, choose one question in a fresh terminal and create an environment inside that directory. Example for Q1:

```bash
cd question_1
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS/Git Bash:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
```

Use a separate environment per question; do not install all four requirements together. Q3 records its owner's exact tested versions and environment limitations. Q2 Kaggle setup preserves the installed CUDA PyTorch stack. Internet is required for fresh dataset/model downloads. No assignment-1 Brown/Treebank/Spanish corpus download step applies to this assignment.

Use a fresh working copy when retraining so generated checkpoints, vocabulary and results do not overwrite the supplied evidence. Successful reproduction need not be bitwise identical across package/hardware versions.

## Running each question

Run the following from the indicated question directory after installing that question's dependencies.

### Q1 - Attention-based English-Hindi translation

```bash
cd question_1
python -u train.py baseline 8 6
python -u train.py attention 8 6
python -m notebook NLP_Group_Assignment_2.ipynb
```

The final number is CPU threads. The submitted implementation trains on CPU. Data preparation uses IIT Bombay English-Hindi; both weights are produced by training before notebook evaluation. The supplied executed notebook, curves and translations can be inspected immediately. The notebook primarily loads and analyzes trained models; review cells before rerunning. Original processed-data caches and optional Q1 weights are excluded from submission.

**Recorded:** 50,000 training, 430 development and 1,723 filtered test pairs; baseline/attention BLEU 0.60/0.92. Both systems are weak; the report states decoding and heuristic-analysis limitations. [Detailed Q1 instructions](question_1/README.md).

### Q2 - Amazon five-class review rating prediction

All six training runs, final evaluation and local CPU finalization have completed. No extra Kaggle run is needed to finish the current experiment.

- [Nine executed Kaggle notebooks and local finalization records](question_2/notebooks/executed/README.md).
- [Detailed staged Kaggle/CLI reproduction guide](question_2/KAGGLE_RUN_GUIDE.md).
- [Six-page Q2 report](question_2/report/Q2_Technical_Report.pdf).
- Results: `question_2/results/`; run histories/logs/environments: `question_2/training_records/`.

For a fresh reproduction, from `question_2/`:

```bash
python tools/package.py --output dist/q2_source.zip
```

Upload the generated source and use the guide's phase sequence: prepare -> smoke -> six model/seed runs -> evaluate -> finalize. Use its exact input prerequisites and accelerator settings. Source templates are in `notebooks/`; the `executed/` subfolder preserves actual outputs. Completed transformer DDP phases use two T4 GPUs; the retained mean-seed42 run used one, as disclosed in the report.

**Recorded two-seed test means:**

| Model | Accuracy | Macro F1 | MAE |
|---|---|---|---|
| GloVe-BiGRU + attention | 63.79% | 0.6351 | 0.4277 |
| RoBERTa mean pooling | 67.13% | 0.6706 | 0.3672 |

Sample SD, per-class scores and rating-distance errors are included. Test results do not guide further tuning. [Detailed Q2 instructions](question_2/README.md).

### Q3 - Abstractive summarization with a scratch Transformer

```bash
cd question_3
python download_data.py --work ../work --train-size 2048 --validation-size 1024
python -m unittest discover -s . -p test_solution.py -v
python train.py --work ../work --output results --epochs 12 --patience 2 --batch-size 16 --threads 4 --lr-scale 0.1
python verify_inputs.py --work ../work
python evaluate.py --work ../work --output results --batch-size 32 --threads 4
python make_report.py
```

The tokenizer is trained only on the declared training subset. The code verifies pinned subset/tokenizer fingerprints. The full 11,490-example test split is used for evaluation. This command block describes reproduction in a working copy, not an instruction to repeat test-driven model selection.

**Recorded:** model ROUGE-1/2/L/Lsum = 1.7506/0.0472/1.6257/1.7042 (0-100 scale), well below Lead-3. Revised outputs are more varied, but repetition and poor summaries remain; the previously evaluated test set is not described as newly blind. Fourteen correctness tests passed locally. Per PDF instructions, Q3 datasets and checkpoints are excluded. [Detailed Q3 instructions/environment](question_3/README.md).

### Q4 - Plot-twist generation with RNN and LSTM

The two included checkpoints and vocabulary allow generation without preprocessing or retraining:

```bash
cd question_4
python -m src.generate --model rnn --strategy greedy
python -m src.generate --model lstm --strategy temperature --temperature 0.8
python -m src.challenge
```

For a fresh training/evaluation reproduction, in a working copy:

```bash
python -m src.preprocess
python -m src.train --model rnn
python -m src.train --model lstm
python -m src.plot_curves
python -m src.metrics
```

Preprocessing downloads TinyStories and builds the 20k/1k/1k story-level splits. Metrics reevaluation requires those processed test sequences. Both model files and `artifacts/vocab.json` must remain a matching set.

**Recorded:** RNN/LSTM test perplexities 16.88/13.40; seven challenge openings and 84 records. Greedy, temperatures 0.5/0.8/1.2 and top-k outputs are included. Neither lower perplexity nor diversity proves better storytelling.

**Required human evaluation remains incomplete.** At least ten generated continuations must receive genuine blinded 1-5 scores for coherence, grammar, relevance and creativity/surprise, with three evaluators wherever feasible. Forms/key/instructions are in `question_4/results/`. Keep `human_eval_key.csv` away from evaluators until scoring finishes. Do not regenerate forms over collected ratings. [Detailed Q4 instructions](question_4/README.md).

## Checkpoints, data and GitHub policy

| Item | Push to GitHub? | Include in final ZIP? |
|---|---|---|
| Source, configs, requirements, measured results/figures, executed notebooks and combined PDF | Yes | Yes |
| `question_4/checkpoints/rnn.pt` and `lstm.pt` | **Yes: explicit ignore exceptions** | **Yes: required by Q4** |
| `question_4/artifacts/vocab.json` | Yes | Yes: required to use the checkpoints |
| Full datasets/processed training splits, environments and caches | No | No |
| Q3 checkpoints | No | No: explicitly excluded by Q3 |
| Large Q2 checkpoints and original phase ZIPs | No | No; retained locally for model preservation |
| `_working_archive/`, temporary files, failed-run backups and duplicate archives | No | No |

Q4's models are about 21.1 MiB and 25.7 MiB. Both fit ordinary Git's current per-file limit; use command-line Git for this repository. The second exceeds GitHub's 25 MiB browser-upload limit. [GitHub file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Root and Q4-specific `.gitignore` files exclude generated data while allowing the two required Q4 models and its data README. Ignoring a file affects Git tracking; the submission ZIP must follow its own inclusion rules. Do not use a Git archive of tracked files until required files have actually been committed.

From repository root, review and push when ready:

```bash
git status --short
git add -A
git diff --cached --stat
git commit -m "Add Assignment 2 implementations, results and combined report"
git push -u origin main
```

The configured remote is `https://github.com/arnavmishra2005/nlp-assignment-2.git`.

## Rebuilding the combined report

```bash
python -m pip install -r report/requirements_report.txt
python report/build_combined_report.py
```

Q1 Hindi text needs a Devanagari TrueType font. On Windows the builder defaults to `C:/Windows/Fonts/Nirmala.ttc`; on another platform pass `--font /path/to/NotoSansDevanagari.ttf`. The PDF embeds the font, so readers do not need it installed. Existing Q2/Q3/Q4 section PDFs are incorporated without retraining. The builder checks section/page counts, records report page counts and section ranges and writes `report/combined_report_check.json`. Rebuild after reviewed section reports change.

## Submission format

Submit one group ZIP containing `README.md`, `CONTRIBUTIONS.md`, `question_1/` through `question_4/`, and `report/Combined_Technical_Report.pdf` with its supporting report files. Include required code, notebooks, measured results, visualizations and Q4 checkpoints/vocabulary. Exclude Git metadata, working archives, full datasets, virtual environments and caches.

Validate the extracted ZIP, confirm that report numbers match the included measurements, and ensure the four question sections and required human-evaluation results are included. Arnav submits the single group ZIP to the correct Classroom assignment before the deadline.

## Troubleshooting

| Problem | Correction |
|---|---|
| `ModuleNotFoundError` | Activate the selected question's environment and install its requirements; do not combine all four environments |
| `No module named src` for Q4 | Run `python -m src...` from `question_4/` |
| Q1 notebook cannot find weights | Train both models first in a working copy; the submitted executed notebook is available for immediate inspection |
| Q2 source/prerequisite mismatch | Use matching verified source/config/data/phase inputs; follow the staged guide rather than bypassing fingerprint checks |
| Q3 input/tokenizer fingerprint mismatch | Use the pinned subset sizes/order and declared tokenizer; do not silently reuse a different workspace |
| Q4 metrics missing `test.pt` | Run preprocessing in a working copy; generation itself only needs supplied models/vocabulary |
| Hindi font unavailable when rebuilding PDF | Pass a compatible Devanagari font using `--font`; the existing PDF can be read without installing one |
| Blank Q4 scores | Obtain genuine ratings; a prepared template is not a completed evaluation |

Original downloads and audit/backup records are retained locally under `_working_archive/` and intentionally absent from the Git/submission contents. See each question's README for experiment-specific details and dataset usage conditions.
