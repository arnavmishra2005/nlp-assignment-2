# Q2 requirement-to-evidence map

| Requirement | Submitted evidence | Status |
|---|---|---|
| Official splits, title/body recovery, preprocessing and train-only vocabulary | config; data/audit code; run manifests; report; executed preparation notebook | Executed and inspected |
| GloVe two-layer BiGRU with additive attention and embedding freeze/unfreeze | models/train code; both baseline notebooks and training records | Both seeds complete |
| Fully fine-tuned RoBERTa with masked mean pooling | model/training code; two mean notebooks and records | Both seeds complete |
| Optimizers, warmup, clipping, precision, seeds and recovery | config and run/complete/environment/training records | Recorded; mixed topology disclosed |
| One ablation, two seeds | first-token notebooks; validation and paired-ablation tables | Both seeds complete; validation comparison |
| Final Accuracy, Macro/per-class F1, MAE and normalized confusion | per-seed prediction/metric files; aggregate tables and figures | Inspector recomputation passed |
| Adjacent versus distant rating errors | final metric tables and six-page report | Included with denominator distinctions |
| Mean and sample SD over >=2 seeds | test/validation/per-class summaries | Recomputed from per-seed evidence |
| Training/validation loss and Macro F1 curves | results/figures; report and executed notebooks | Included |
| >=5 baseline-wrong/mean-right and >=2 reverse if available | raw examples; separate annotated CSV; report pages 5-6 | Five forward and two reverse included |
| Token-level importance | scores, method metadata and figure; report interpretation | Included; causal limits disclosed |
| Concise report <=6 pages including figures/tables | report/Q2_Technical_Report.pdf and page-check JSON | Exactly six pages |

See VERIFICATION.md for scope. Counts refer to measured experiments; no numerical target or independent robustness claim is implied.
