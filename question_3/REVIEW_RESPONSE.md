# Q3 review investigation

The original failure is **learned encoder representation collapse**. No defect
was found in cached decoding, source-memory routing or padding/causal masks in
the checks below. This conclusion is bounded by the tested paths, not a claim
that testing proves every possible input correct.

## Evidence from the original checkpoint

- Six fixed real validation articles (indices 0, 1, 2, 100, 500 and 1000) have
  different tokenized sources. All source-token hashes are recorded.
- After encoder layer 1, cross-article RMS differences are about 0.86–0.87.
  After layer 2 they are about 0.0013. Fresh initial weights produce about
  1.36–1.37. The source representations have become almost constant through
  learning, despite different inputs.
- Both decoder layers receive the expected projected encoder keys/values and
  exact source mask on every generation step. Masked positions receive zero
  attention, including a validation example shortened to exercise padding.
- Cached and uncached greedy IDs match exactly on the six articles. Independent
  per-article generation matches batched generation. Full-prefix versus cached
  logits differ by at most 5.73e-6 over a 20-token real prefix.
- The complete 1,024-example validation loss recomputes to
  **6.020055199688513**, exactly the checkpoint's saved value.
- Shuffling the six encoder memories changes teacher-forced loss by only about
  1.4e-6. Zeroing the memory changes logits, confirming that the decoder memory
  path exists, but the learned differences among real sources are negligible.
- Gradients are finite and reach the encoder, decoder, embeddings and output.

These checks are reproducible with `audit_collapse.py`; the original evidence is
in `results/audit/diagnostics.json`. Original weights remain outside the ZIP, as
required by the assignment.

## Training stability revision

The original Noam-style schedule peaks at **0.0088388** after 200 updates for
width 64. This is aggressive for the small post-normalization model. The revised
run changes only the learning-rate scale to **0.1**, with peak **0.00088388**,
and allows at most 12 epochs with patience 2. Architecture, initialization seed,
data order/subsets and tokenizer remain the same. The best checkpoint is selected
within that fixed run using validation loss. The original encoder collapse and
repetitive output were independently reproduced on validation data before this
decision. No test ROUGE is used to select the revision.

The revised audit found **6/6 different validation outputs**, final encoder
cross-article RMS differences of **1.43–1.46**, and exact cached/uncached greedy
agreement. Recomputed validation loss is **6.182745431360581**, exactly matching
the selected epoch-12 checkpoint. The encoder collapse is removed in these
checks; highly repetitive, poor-quality generation remains. Shuffling source
memories now raises the six-example teacher-forced loss from about 6.2255 to
6.2790, unlike the negligible change in the original model.

The revised audit and recomputed loss are in `results/audit_revised/`. The final
report and `results/metrics.json` describe the revised checkpoint. A lower
teacher-forced loss does not guarantee useful generation; residual repetition,
omissions and unsupported facts are explicitly retained in the analysis. The
previously evaluated test set is reused for the final benchmark, so this is not
a new blind evaluation.

## Reused data and tokenizer

Fresh streams from pinned CNN/Daily Mail 3.0.0 repository revision
`96df5e686bee6baa90b8bee7c28b81fa3fa6223d` matched all existing train, validation
and test subset contents. Rebuilding SentencePiece exclusively from the declared
2,048 training pairs reproduced every vocabulary piece and score, normalization,
4,096 vocabulary size and PAD/UNK/BOS/EOS IDs 0/1/2/3.

The original reuse happened to be valid, but silently trusting reused files was
a real reproduction weakness. `download_data.py`, `train.py`, and `evaluate.py`
now check the bundled semantic manifest. Changed row order/content, wrong subset
sizes, duplicate/overlapping IDs and a different tokenizer are rejected.

## Actual environment and reproduction checks

Python **3.14.8**, Windows 11, CPU PyTorch runtime **2.14.1+cpu**, distribution
metadata **2.14.1**. The unusual recorded PyTorch version was confirmed directly
from the live imported module and package metadata, rather than inferred from
the report. Original training used NumPy 2.5.3; revised training uses the local
2.5.4 overlay. Other observed direct versions are pinned in `requirements.txt`.

A separate virtual environment reusing the host CPU PyTorch installed the pinned
requirements, passed `pip check`, passed all **14** tests, and ran data/tokenizer
verification and the downloader's offline reuse path successfully. See
`results/audit/reproduction_tests.txt` and `dependency_check.txt`. The original
12 tests also passed again before adding the two fingerprint tests. An isolated
installation on a completely clean machine is not claimed. README commands state
the tested environment and the limits of numerical reproducibility.
