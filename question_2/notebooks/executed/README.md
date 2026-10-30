# Executed Q2 experiment notebooks

These are the actual downloaded Kaggle executions, renamed and moved without changing their bytes, cell outputs or execution counts. Each selected notebook has three executed code cells, no saved error output, a phase-completion marker and a final inspection receipt matching the retained result archive fingerprints.

| Order | Notebook | Phase |
|---|---|---|
| 01 | [01_prepare_data_CPU_executed.ipynb](01_prepare_data_CPU_executed.ipynb) | prepare |
| 01b | [01b_smoke_GPU_executed.ipynb](01b_smoke_GPU_executed.ipynb) | smoke |
| 02 | [02_baseline_seed42_executed.ipynb](02_baseline_seed42_executed.ipynb) | baseline_seed42 |
| 03 | [03_baseline_seed123_executed.ipynb](03_baseline_seed123_executed.ipynb) | baseline_seed123 |
| 04 | [04_roberta_mean_seed42_executed.ipynb](04_roberta_mean_seed42_executed.ipynb) | roberta_mean_seed42 |
| 05 | [05_roberta_mean_seed123_executed.ipynb](05_roberta_mean_seed123_executed.ipynb) | roberta_mean_seed123 |
| 06 | [06_roberta_first_seed42_executed.ipynb](06_roberta_first_seed42_executed.ipynb) | roberta_first_seed42 |
| 07 | [07_roberta_first_seed123_executed.ipynb](07_roberta_first_seed123_executed.ipynb) | roberta_first_seed123 |
| 08 | [08_final_evaluation_and_report_executed.ipynb](08_final_evaluation_and_report_executed.ipynb) | evaluate |

## CPU finalization

Finalization ran locally using the original CPU launcher. It has JSON execution/inspection evidence instead of an executed Kaggle notebook. The unexecuted `../09_finalize_CPU.ipynb` is a reproduction template, not a record of a Kaggle job. The copied `09_finalize_CPU_execution.json` and `09_finalize_CPU_inspection.json` record the actual local completion.

## Submission and provenance

Include all nine selected notebooks and this inventory in the group submission. Keep the source templates in the parent notebooks directory for reproduction. An older baseline-seed42 execution failed only during output delivery; its complete original notebook is retained separately by the coordinator and is not selected as final experiment evidence. No additional training is needed.

The current Q2 folder contains the selected executed notebooks and the finished six-page report in `../../report/`. Include this folder during group submission assembly; do not substitute an earlier phase ZIP that predates these additions.
