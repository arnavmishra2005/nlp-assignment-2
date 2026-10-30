# NLP Group Assignment 2 - Contribution Split

## Team Members

1. **Vatsal Yadav** - B24DS036
2. **Rudra Mehul Dudhat** - B24DS506
3. **Mohak Arya** - 12341420
4. **Kanishk Nandeshwar** - B24DS010
5. **Arnav Mishra** - 12340330

---

## Question 1: Attention-Based English-Hindi Translation - **Rudra Mehul Dudhat**

| Member | Responsibility | Deliverables |
|---|---|---|
| **Rudra Mehul Dudhat** | Q1 implementation and experiments | `question_1/data.py` - IIT Bombay corpus loading, cleaning, tokenization, vocabularies and padding; `models.py` - baseline LSTM encoder-decoder and inspectable additive attention; `train.py` - comparable teacher-forced training and validation-based checkpoint selection |
| **Rudra Mehul Dudhat** | Q1 evaluation and analysis | `question_1/analyze.py`, executed `NLP_Group_Assignment_2.ipynb`, `results/` - translations, BLEU/word-overlap and teacher-forced accuracy, learning curves, sentence-length analysis and six attention/error cases; `results_run1_30k/` - historical measured comparison evidence |
| **Rudra Mehul Dudhat** | Q1 documentation | `question_1/REPORT.md`, reproduction README and dependency list |

---

## Question 2: Amazon Review Rating Prediction - **Arnav Mishra**

| Member | Responsibility | Deliverables |
|---|---|---|
| **Arnav Mishra** | Q2 implementation and reproducible workflow | `question_2/q2/data.py`, `audit.py`, `models.py`, `train.py`, `train_distributed.py`, `workflow.py` - title/body recovery, preprocessing, GloVe-BiGRU attention baseline, fully fine-tuned RoBERTa, distributed execution, checkpoints and staged execution; `config.json` and launch/setup scripts |
| **Arnav Mishra** | Q2 training, ablation and evaluation | Six complete training runs across seeds 42/123, mean-versus-first-token ablation, frozen final test evaluation; `question_2/results/` - Accuracy, Macro/per-class F1, MAE, confusion matrices, ordinal errors, predictions, figures and token importance; `training_records/` - run metadata/histories/logs |
| **Arnav Mishra** | Q2 experiment evidence and documentation | Nine successful Kaggle executions under `question_2/notebooks/executed/`, local CPU-finalization evidence, `question_2/report/Q2_Technical_Report.pdf` (six pages), report builder and separate annotated qualitative examples; reproduction README/guide |
| **Arnav Mishra** | Group integration and submission coordination | Folder organization, retained provenance, cross-question checks and contribution record; responsible for assembling one combined report and one group ZIP and making the final submission |

---

## Question 3: Transformer Summarization from Scratch - **Vatsal Yadav**

| Member | Responsibility | Deliverables |
|---|---|---|
| **Vatsal Yadav** | Q3 implementation and training | `question_3/data.py`, `download_data.py`, `verify_inputs.py`, `model.py`, `train.py` - training-only shared BPE, pinned data/input verification, elementary Transformer encoder-decoder/attention/masks and validation-selected training |
| **Vatsal Yadav** | Q3 evaluation, diagnosis and revision | `evaluate.py`, `audit_collapse.py`, `test_solution.py`, `results/` - full-test predictions, four ROUGE metrics and Lead-3, generation attention, five examples, original/revised collapse diagnostics and revised learning-rate experiment |
| **Vatsal Yadav** | Q3 documentation | `Final_Report.pdf`, `make_report.py`, `REVIEW_RESPONSE.md`, README, dependency versions and experimental limitations |

---

## Question 4: Plot-Twist Generator - **Mohak Arya & Kanishk Nandeshwar**

| Member | Responsibility | Deliverables |
|---|---|---|
| **Mohak Arya & Kanishk Nandeshwar** | Joint Q4 implementation and experiments | `question_4/src/data.py`, `preprocess.py`, `models.py`, `train.py`, `config.py` - TinyStories preparation, shared vocabulary, comparable vanilla RNN/LSTM next-token training; `checkpoints/rnn.pt`, `checkpoints/lstm.pt`, `artifacts/vocab.json` and training histories |
| **Mohak Arya & Kanishk Nandeshwar** | Joint generation and quantitative evaluation | `src/generate.py`, `challenge.py`, `metrics.py`, `plot_curves.py`; greedy/three-temperature/top-k continuations, seven-opening challenge with 84 records, perplexity/diversity/repetition/length results and loss curves |
| **Mohak Arya & Kanishk Nandeshwar** | Joint human-evaluation preparation and report | `src/human_eval.py`, blinded samples, separate answer key and rating forms; `Plot_Twist_Generator_Final_Report.pdf` and editable DOCX; human-evaluation design and reporting |

---

## Contribution Summary

| Member | Primary question | Documentation / additional responsibility |
|---|---|---|
| **Rudra Mehul Dudhat** | Q1 | Q1 report, experiment interpretation and reproduction instructions |
| **Arnav Mishra** | Q2 | Q2 report/evidence, group integration and final submission responsibility |
| **Vatsal Yadav** | Q3 | Q3 report, collapse investigation, revision and reproduction evidence |
| **Mohak Arya** | Q4, jointly | Q4 experiments, evaluation and documentation jointly with Kanishk |
| **Kanishk Nandeshwar** | Q4, jointly | Q4 experiments, evaluation and documentation jointly with Mohak |
