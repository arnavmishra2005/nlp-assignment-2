# Q2 Kaggle run guide: prepare -> check -> run one phase -> inspect -> continue

This guide supersedes the earlier eight-notebook instructions. The models, official data splits and experiments are unchanged. The new workflow adds explicit audits, a separate correctness smoke phase, three-cell notebooks, phase archives, inspections and CPU finalization.

## Current phase: roberta_first_seed123 (sixth and final full training run)

**First-token seed42 passed full inspection on 30 October 2026.** All18 inventory hashes, exact source/config/prepared/split fingerprints, both active T4s/NCCL and gradient-check evidence match. Three epochs, selected epoch3; validation accuracy0.6678, Macro F1 0.6665230252848238, MAE0.3736. Training12,812.5seconds (3h33m32s), approximately71 minutes/epoch. All saved train/validation metrics were recomputed from confusion matrices. Test remains untouched. Five of six full training experiments are verified.

1. Upload root-level **Q2_roberta_first_seed42_results.zip** (454.6 MB) as a private Kaggle dataset named **q2_roberta_first_seed42**.
2. Import root-level **07_roberta_first_seed123.ipynb** into a fresh notebook named **NLP Assignment 2 Q2 RoBERTa First Seed123 Dual GPU**.
3. Attach exactly **q2_source** (reuse the current version from the successful run), **q2_prepare_results**, and **q2_roberta_first_seed42**. **No source update is needed.** Keep every successful training archive for the final evaluation phase.
4. Select **GPU T4 x2** and **Internet On**. Leave `PHASE = 'roberta_first_seed123'` and supplied path overrides unchanged (`DATA = None`, `PREVIOUS = []`, `RESUME_FROM = None`, `SOURCE_PATH = None`, `GLOVE_PATH = None`).
5. Choose **Save Version -> Save & Run All**. Require SOURCE_VERSION_CHECK_PASSED, DUAL_GPU_SETUP_PASSED, PREREQUISITES_VERIFIED, then DUAL_GPU_COMMUNICATION_PASSED, DDP_GRADIENT_CHECK_PASSED and DDP_TRAINING_ENABLED. Training must identify model=roberta_first seed=123.
6. This is the second independent seed of the required first-token pooling ablation: full200,000 train/5,000 validation, pinned original pretrained weights, seed123, same optimizer and settings, local8 x2GPUs xaccumulation2 = effective batch32. Maximum3 epochs with validation early stopping. The preceding checkpoint is verified as a prerequisite; this run does not continue its weights.
7. Allow approximately3h30-3h45 if all three epochs complete at the measured speed, plus any unusual setup delay; early stopping can shorten it. Download **/kaggle/working/Q2_roberta_first_seed123_results.zip** individually after PHASE_COMPLETE and INSPECTION_PASSED, and return it for inspection.

This is the last training run. After its archive passes inspection, run final evaluation and analyses, then finalization/report editing and group submission assembly. Final evaluation needs prepared data plus all six successful training archives and the same source dataset; do not discard earlier outputs. Current upload files are beside the main README and graphical bars remain disabled. Notebook07 is byte-identical to its copy in the already uploaded source ZIP.

## Two-GPU experiment preservation

The three remaining RoBERTa runs use one DDP worker per T4. The completed mean-pooling seed42 run retains its original single-GPU execution. The config JSON is byte-for-byte unchanged: same pretrained revision, full official data, tokenization/maximum length, architectures, pooling, dropout probability, seeds, effective batch32, learning rate, AdamW groups/decay, 8% warmup, scheduler update count, clipping, AMP policy, epoch caps and validation selection. The same seeded global training permutation is sliced without replacement between ranks; no extra duplicated samples or dropped rows are introduced. Summed local losses are multiplied by world size before dividing by the original actual accumulation-group sample count, compensating for DDP gradient averaging. `no_sync` spans the first accumulation forward/backward; synchronization happens on the optimizer-boundary microbatch.

Two-GPU reductions and per-rank dropout streams change the numerical trajectory. Exact weights/scores are not promised to match single-GPU runs, and speedup must be measured. This hardware execution change is recorded in `distributed_execution.json`, `run.json`, completion evidence, per-rank peak memory and startup all-reduce/device records. Validation still evaluates every official validation row once. Best checkpoints contain ordinary unwrapped parameter names, so existing prediction, analysis, evaluation and packaging interfaces remain compatible. Recovery stores optimizer/scheduler/scaler plus each rank's RNG and partial-epoch statistics, with the same world size required for resumption. Training and validation DataLoader workers use spawn, avoiding fork with NCCL.

The old completed baselines and prepare/smoke archives remain accepted only for their known source fingerprints and pinned unchanged data/model/single-GPU trainer code. The preflight print-message correction is separately pinned. Original provenance/hashes are retained; the specifically inspected original single-GPU mean-seed42 run is retained at the user's request. Other old single-GPU transformer phases are not automatically accepted. Remaining runs use the same new source, and the mixed topology is disclosed as a limitation of pooling/seed-only attribution.

References: [PyTorch DDP gradient averaging and no_sync](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html), [numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

## Output size and assignment preservation

The storage update changes **exports only**, not data, architectures, seeds, epoch caps, learning rates, precision during training, model selection, metrics, plots or report requirements. Model weights retain their exact original tensor values and dtypes; no quantization, pruning, shortened training or dataset reduction is used.

- Source, logs and intermediate artifacts use **`/kaggle/temp/`**, outside Kaggle's saved output directory. After inspection, `deliver_phase.py` publishes only **one project ZIP directly under `/kaggle/working/`**. Logs/config/manifests remain inside it. Failed phases attempt to publish one `_FAILED.zip` with diagnostic/recovery evidence. Hard termination can prevent failure packaging; temporary files are not guaranteed to survive a killed job.
- Completed smoke checkpoints retain exact model weights/configuration but omit their no-longer-needed optimizer/scheduler/scaler/RNG state. Every model tensor is reloaded and checked for exact equality before replacement. A storage audit records original/new hashes and sizes.
- Full training still writes complete live `last.pt` recovery snapshots at the original frequency. **Failed/unfinished phase archives retain available full recovery state.** A successful completed training archive retains the exact best checkpoint plus full histories/logs/config/evidence; it omits the redundant final optimizer recovery snapshot because later phases do not continue that completed optimization.
- Evaluation keeps results, predictions, histories and exact checkpoint-hash references to the retained training archives, instead of duplicating all six models. **Keep the six successful training archives** for model reuse/reproduction.
- Fast **lossless ZIP compression** is applied to tensors and arrays. The final phase exports a single flat archive containing `question_2/` source/results and phase evidence; there is no nested submission ZIP to download separately.
- Original prepare/smoke archives remain accepted only through a narrow compatibility check: their exact original source ID and every unchanged preprocessing/model/training/smoke core file hash must match. Original provenance is retained and compatibility is explicitly reported. Arbitrary source changes remain rejected.

Transformer weights are intrinsically large. Successful RoBERTa archives will still be hundreds of MB; reducing them to a few MB would require omitting or changing the model, which this workflow does not do.

## The three notebook cells

- **Cell 1:** phase and explicit-path overrides; Python/package versions, nvidia-smi, framework CUDA availability/name, a small GPU matrix operation for GPU phases, free disk. CPU phases reject an allocated GPU to avoid wasting quota.
- **Cell 2:** locate source and copy code to `/kaggle/temp/q2-code/question_2`; check compatibility before installing only missing/incompatible non-CUDA dependencies; preserve the installed PyTorch/CUDA stack. Verify prerequisite archives and restore needed artifacts. Print the complete six-run plan and setup markers. No model training occurs here.
- **Cell 3:** launch `kaggle_start.py` with the current interpreter and unbuffered output. Stream stdout/stderr, provide quiet-period heartbeat messages, check the exit code and inspect the archive. Print success only after all required evidence passes.

A heartbeat means the subprocess is alive; it does not prove useful progress. No persistence setting or previous interactive notebook state is required. Saved source, inputs, checkpoints and phase archives carry the project between fresh sessions.

## Full phase plan (run only after the preceding output has been inspected)

Every phase attaches the same source dataset. In the table, `prepare` means the saved output containing `Q2_prepare_results.zip`; attach saved notebook outputs through Add Input, or upload the returned result ZIP as a private dataset. Both zipped and automatically extracted phase archives are supported.

| Phase in cell 1 | Other inputs to attach | Accelerator | Result archive |
|---|---|---|---|
| `prepare` | None; optional official raw/GloVe inputs | None | `Q2_prepare_results.zip` |
| `smoke` | prepare | GPU | `Q2_smoke_results.zip` |
| `baseline_seed42` | prepare + smoke | GPU | `Q2_baseline_seed42_results.zip` |
| `baseline_seed123` | prepare + baseline_seed42 | GPU | `Q2_baseline_seed123_results.zip` |
| `roberta_mean_seed42` | prepare + baseline_seed123 | GPU T4 x2 | `Q2_roberta_mean_seed42_results.zip` |
| `roberta_mean_seed123` | prepare + roberta_mean_seed42 | GPU T4 x2 | `Q2_roberta_mean_seed123_results.zip` |
| `roberta_first_seed42` | prepare + roberta_mean_seed123 | GPU T4 x2 | `Q2_roberta_first_seed42_results.zip` |
| `roberta_first_seed123` | prepare + roberta_first_seed42 | GPU T4 x2 | `Q2_roberta_first_seed123_results.zip` |
| `evaluate` | prepare + **all six** completed training outputs | GPU | `Q2_evaluate_results.zip` |
| `finalize` | prepare + evaluate | None | `Q2_finalize_results.zip` (single flat final Q2 archive) |

Set **Internet=On** for dependency setup. The four RoBERTa training phases require **T4 x2** and launch two DDP workers. Baseline/smoke/evaluate keep their original execution paths. Training uses cached prepared assets; reserved test is downloaded only after freezing all six runs. Preparation/finalization use CPU.

You can reuse `00_kaggle_workflow.ipynb` by changing only `PHASE` for the current stage and attaching exactly the inputs in the table. The other notebooks have those phase names prefilled and exactly the same three code cells. Do not run both the generic and prefilled notebook for the same experiment.

Source/dataset/previous paths normally auto-discover. If ambiguous, set:

```python
SOURCE_PATH = '/kaggle/input/.../q2_source.zip'  # or extracted source folder
DATA = '/kaggle/input/.../Q2_prepare_results.zip'  # any phase after prepare
PREVIOUS = ['/kaggle/input/.../Q2_smoke_results.zip']  # example for baseline_seed42
```

For `evaluate`, `PREVIOUS` can explicitly list all six training archives. Paths must match the actual Add Input tree, including nested owner/dataset directories. Never select an arbitrary candidate when several versions exist.

## Experiments and quota

There are **six full training runs**, seeds **42 and 123**:

- Two BiGRU + GloVe + attention runs, each up to **8 epochs**.
- Two RoBERTa mean-pooling runs, each up to **3 epochs**.
- Two first-token-pooling ablation runs, each up to **3 epochs**.

That is **28 maximum epochs across runs**, not one eight-epoch run for the whole project. Validation Macro F1 early stopping can shorten runs. No numerical accuracy target is specified; a completed experiment can legitimately perform worse than another model.

`smoke` is separate: two training batches per main model, validation on two examples per class, checkpoint save/reload, predictions and metrics. It never uses test and its outputs say **SMOKE_ONLY_NOT_ASSIGNMENT_RESULTS**. It is a correctness gate, not a hyperparameter trial or a promised runtime benchmark.

Thirty GPU hours is the available budget, not a guaranteed runtime. Logs report actual epoch durations, elapsed time, batch progress and approximate remaining work for the active run. Each successful archive carries accumulated `timing_evidence.json` and `remaining_time_estimate.json` from earlier stages. To review:

```powershell
python estimate_runtime.py "PATH_TO/Q2_roberta_mean_seed42_results.zip" --overhead-minutes 15
```

Estimates distinguish model speeds, remaining capped epochs, early stopping, timing variability and assumed metadata/evaluation/export overhead. They explicitly report unknown model timings; they do not estimate the entire pipeline from a lone baseline epoch. If first-token timings are unavailable, observed mean-pooling encoder timings may be a clearly labeled proxy with a wider range. Any clock estimates use **Asia/Calcutta / UTC+05:30**. Hardware changes may invalidate them. Check your account's actual session limits and remaining quota; do not assume fixed limits from an old project.

## Audits and data policy

Preparation validates IDs, labels, counts, schema and title/body recovery. It saves `prepared/audit/audit.json`, `split_manifest.csv`, `split_policy.json`, near-duplicate candidates and label-conflict candidates. It reports missing titles/bodies, class balance, exact/normalized duplicate text, cross-split overlaps and available product/reviewer group sizes. Normalization proposes near-duplicate candidates; actual character similarity is checked. This bounded diagnostic is not exhaustive semantic duplicate detection; no transitive grouping or automatic deletion occurs.

The PDF explicitly requires the corpus's existing 200,000/5,000/5,000 splits. Keep them, disclose any shared products/reviewers/texts, and do not silently create different splits. Vocabulary fits training only. Validation selects checkpoints and stopping. Test is checked/evaluated only in the frozen final evaluation phase. Prepared-file inventories, dataset/model revisions, class mapping and split fingerprints must agree throughout.

## Inspection and archives

The launcher inspects prerequisites before full training and automatically inspects each result ZIP before announcing success. Return each ZIP for review before moving to the next stage. The inspector checks ZIP/inventory integrity, status, source/config/data fingerprints, expected experiment/seed counts, checkpoints, complete histories, chosen epochs, sample IDs, finite probabilities in [0,1], row sums, argmax labels, per-seed metric agreement, aggregate mean/sample-SD tables, figures and flat final submission file integrity.

Every successful phase writes `execution.json` with **COMPLETE** only after the archive passes. Other states are **PENDING**, **RUNNING**, **FAILED**. A best checkpoint filename alone is not a completion signal. Reused completed runs verify history, configuration and evidence hashes.

Archives contain the evidence needed by the next phase: preparation includes caches/vocabulary/initial weights; successful training includes the exact best checkpoint and full records, while failed/unfinished training archives include available rolling optimizer/scheduler/scaler/random-state recovery checkpoints. Evaluation includes final evidence plus exact checkpoint references to the retained training archives. Fast lossless compression reduces size without changing tensor values. Allow at least **8 GiB free working disk** for data/results plus their archive; the default preflight checks this. Generated extracted prerequisite caches are removed after successful packaging; reproducibility uses saved inputs, not retained session state.

## Recovery and failures

On failure, the launcher preserves diagnostics and attempts `Q2_<phase>_FAILED.zip`. It is **recovery-only**, never a valid completed prerequisite. Inspect it explicitly with:

```powershell
python inspect_results.py "PATH_TO/Q2_baseline_seed42_FAILED.zip" --expected-phase baseline_seed42 --allow-incomplete
```

To resume, start the same phase with its original prerequisites, attach that failed archive and set `RESUME_FROM` in cell 1 to its exact path. The launcher verifies its integrity/config/source/data fingerprint, restores your run and resumes from `last.pt` when available. Keep `best.pt`, `last.pt`, histories and run metadata together. Recovery can lose batches since the last checkpoint. Failed preparations without a completed download may need to download again.

Kaggle must have retained the files. Hard cancellation/termination can prevent archive creation; if a valid phase manifest exists in retained output, an extracted phase directory can be inspected/restored. Temporary files outside the saved output directory may be lost after hard termination; do not assume a file survived just because a run once displayed a checkpoint message. Save & Run All is background execution; closing the browser is different from cancelling the job.

Nonzero exit code, failed prerequisite, missing archive, invalid predictions or FAILED status means **stop and inspect**, not continue expensive phases. Send the failed archive/log. Do not retune after seeing test results. Avoid editing source/config between runs: source fingerprints intentionally reject mixing versions. If an OOM requires changing microbatch size, revise a single config consistently and rebuild comparable runs before evaluation. Do not silently change batch size or fall back to random embeddings.

## Equivalent terminal commands

From `question_2/` in a compatible environment with CUDA-enabled PyTorch:

```bash
python setup_environment.py --install-missing
python -u kaggle_start.py --phase prepare --output artifacts
python inspect_results.py artifacts/prepare/Q2_prepare_results.zip --expected-phase prepare
python -u kaggle_start.py --phase smoke --data artifacts/prepare/Q2_prepare_results.zip --output artifacts
python -u kaggle_start.py --phase baseline_seed42 --data artifacts/prepare/Q2_prepare_results.zip --previous artifacts/smoke/Q2_smoke_results.zip --output artifacts --seed 42 --resume
```

For subsequent phases use the table's phase name and predecessor archive; repeat `--previous` for the six evaluation inputs. `--inputs` specifies a root to search instead of explicit archives. `--setup-only` performs prerequisites/environment checks without training; `--resume-from` restores your same-phase recovery archive; `--no-resume` rejects an existing rolling checkpoint. `--allow-cpu` exists only for developer synthetic tests, not recommended for full training.

Optional CPU raw-data preparation for later attachment:

```bash
python download_data.py --config config.json --output data/raw
python -u kaggle_start.py --phase prepare --data data/raw --output artifacts
```

This only downloads train/validation raw files. `GLOVE_PATH` / `--glove-path` can point to an attached official GloVe 6B ZIP or matching 100d text file. Test remains reserved. KaggleHub is not required here; Add Input is used for source and preceding outputs. If adding KaggleHub later, use the returned resource path rather than guessing its cache/input location, as described in its official documentation.

## After final evaluation

Run `finalize` on CPU after the evaluate archive is inspected. Download `/kaggle/working/Q2_finalize_results.zip`. It directly contains `question_2/` source, configs, notebooks, result tables/figures, example texts, report draft and training records, alongside phase inspection evidence. There is no inner ZIP to download separately. Full datasets, caches and Q2 model checkpoints are excluded from the submission ZIP; exact selected model checkpoints remain in the six successful training archives; failed/unfinished archives retain recovery state when available. Q2 does not require a demo or external evaluation. The optional `predict.py` CLI is provided, but no deployment/calibration/robustness target is claimed.

Download the **executed Kaggle notebook versions** too; the source ZIP's notebooks are intentionally unexecuted. Review/annotate qualitative examples and the importance visualization, remove TODOs, and format the Q2 report section to **at most six pages**. Integrate it with the other questions into **one combined group technical report** and one group ZIP. `FINAL_PACKAGE_CHECKED.json` deliberately records `human_report_complete=false` until that work is done. Packaging completion is not assignment completion.

## References

- [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks): background saved execution and environment behavior.
- [Official KaggleHub documentation](https://github.com/Kaggle/kagglehub): attached resources inside Kaggle versus local cache outside.
- [Dataset](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en), [title mirror](https://huggingface.co/datasets/buruzaemon/amazon_reviews_multi/tree/main/en), [official GloVe mirror](https://huggingface.co/stanfordnlp/glove).
