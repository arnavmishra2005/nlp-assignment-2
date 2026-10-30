"""Assignment 2 Question 2 experiment package."""
import os

# Set before tqdm, datasets or hub imports, including command-line module runs.
# Training keeps its existing periodic plain-text metrics and epoch summaries.
for _name, _value in {
    "TQDM_DISABLE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    "PIP_PROGRESS_BAR": "off",
}.items():
    os.environ[_name] = _value
