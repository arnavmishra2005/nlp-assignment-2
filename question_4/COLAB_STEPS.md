# Google Colab steps

## 1. Create a notebook and select GPU

1. Open https://colab.research.google.com/
2. Select **New notebook**.
3. Choose **Runtime → Change runtime type → T4 GPU → Save**.
4. Run:

```python
!nvidia-smi
```

## 2. Upload and unzip the repository

Download `plot-twist-generator-colab.zip` from this chat. In a Colab cell:

```python
from google.colab import files
uploaded = files.upload()
```

Choose the ZIP file, then run:

```python
!unzip -q plot-twist-generator-colab.zip
%cd /content/plot-twist-generator-colab
!ls
```

The folder should contain `src`, `requirements.txt`, and `README.md`.

## 3. Install dependencies

```python
!pip install -q -r requirements.txt
```

## 4. Preprocess TinyStories

```python
!python -m src.preprocess
```

Expected output includes a vocabulary near the size in your existing local run, plus train/validation/test statistics. The streamed dataset order can differ from an earlier run, so verify the reported statistics.

## 5. Train both models

Run each command in a separate cell:

```python
!python -m src.train --model rnn
```

```python
!python -m src.train --model lstm
```

## 6. Plot loss curves

```python
!python -m src.plot_curves
from IPython.display import Image, display
display(Image("artifacts/loss_curves.png"))
```

## 7. Generate sample continuations

```python
!python -m src.generate --model rnn --strategy greedy
!python -m src.generate --model rnn --strategy temperature --temperature 0.5
!python -m src.generate --model rnn --strategy temperature --temperature 0.8
!python -m src.generate --model rnn --strategy temperature --temperature 1.2
!python -m src.generate --model rnn --strategy top_k --top_k 10
```

Repeat with `--model lstm`. Outputs are saved under `results/`.

## 8. Run challenge and evaluation

```python
!python -m src.challenge
!python -m src.metrics
!python -m src.human_eval
```

Use at least 3 raters for the blinded CSV. Do not edit generated outputs. Keep the mapping from sample IDs to model and decoding strategy separately for analysis.

## 9. Persist files to Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!mkdir -p /content/drive/MyDrive/plot-twist-generator
!cp -r src artifacts checkpoints data results README.md requirements.txt .gitignore COLAB_STEPS.md /content/drive/MyDrive/plot-twist-generator/
```

Run this copy command again after generating new results.

## 10. Export a ZIP

```python
%cd /content
!zip -qr plot-twist-generator-results.zip plot-twist-generator-colab
from google.colab import files
files.download("/content/plot-twist-generator-results.zip")
```
