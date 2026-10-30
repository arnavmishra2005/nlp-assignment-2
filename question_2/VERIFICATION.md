# Q2 verification and scope

- All six full training runs completed; histories and validation-selected checkpoint hashes are recorded per run.
- The returned final evaluation archive passed the official inspector for all 89 evidence files, including probability/ID checks, prediction-derived metrics, seed means/sample SD, paired ablation and figures.
- Local CPU finalization passed the official inspector for 229 original phase-export files. It did not retrain models or repeat test evaluation.
- Nine successful executed Kaggle notebooks cover preparation, smoke, all six training runs and final evaluation. Every selected notebook's final inspection receipt matched its retained archive source/config/split/prepared-data fingerprint. Original bytes and outputs were preserved during renaming.
- Local finalization is represented by actual execution/inspection JSONs in `notebooks/executed/`; its source notebook remains an unexecuted reproduction template.
- The report builder rendered `report/Q2_Technical_Report.pdf` at exactly six pages. Five forward and two reverse examples were annotated in a separate CSV; the raw selected examples remain unchanged.

The experimental source modules/configuration were not changed by folder cleanup. Earlier synthetic developer-test evidence and original archives are preserved by the coordinator, outside the submission folders. This verification does not claim cross-domain robustness, calibration, significance with only two seeds or unmeasured model behavior.
