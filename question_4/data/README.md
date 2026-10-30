# Dataset files

Raw TinyStories data and processed token-ID splits are intentionally not included in this submission archive. Run `python -m src.preprocess` from the project root to download the selected TinyStories stream subset and recreate `data/processed/{train,val,test}.pt`. An internet connection is required. The saved vocabulary and recorded data statistics remain in `artifacts/`.
