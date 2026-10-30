# Question 1 - English-to-Hindi translation

English-to-Hindi translation using an LSTM encoder-decoder baseline and an additive-attention variant.

Read `REPORT.md` and executed `NLP_Group_Assignment_2.ipynb`. Current results are in `results/`; `results_run1_30k/` is historical comparison evidence. The current run uses 50,000 training pairs, 430 development pairs, 1,723 test pairs, 10,000-entry language vocabularies, seed 0 and eight epochs per model. Reported corpus BLEU (0-100): baseline 0.60; attention 0.92. Retain the weak-performance limitations.

## Reproduce

Use a separate environment. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -u train.py baseline 8 6
python -u train.py attention 8 6
python -m notebook NLP_Group_Assignment_2.ipynb
```

Training uses CPU tensors; the last positional argument sets CPU threads. `data.py` prepares the IIT Bombay English-Hindi corpus (`cfilt/iitb-english-hindi`) and processed cache, downloading when needed. `analyze.py` provides functions; it is not a standalone evaluation launcher. The notebook contains evaluation, figures and error analysis, as well as training commands: review its cells before executing all to avoid duplicate training. Run in a copy if preserving received results unchanged.

Exact original-run package versions and dataset revision were not pinned; numerical reproduction across environments may differ. This submission excludes processed-data caches and optional Q1 weights. For a fresh executable rerun, train both models with the commands above before running notebook evaluation; the supplied executed notebook and measured outputs can be inspected immediately. The coordinator retains original caches/weights outside the submission. Only load trusted pickle/checkpoint files.

The report discloses greedy decoding with UNK suppressed. Preserve this policy/disclosure. Historical versus current runs change data size, epochs, vocabulary and batch size together; the comparison cannot isolate a data-size effect.

Exclude processed dataset `.pkl` caches, environments and incoming ZIPs from submission. Q1 weights are optional under the PDF and are retained outside the submission, rather than distributed without their cached vocabulary. Keep the executed notebook, measured outputs, figures and error analysis.
