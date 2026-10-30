# Combined technical report

**[Combined_Technical_Report.pdf](Combined_Technical_Report.pdf)** is the actual assembled 30-page report covering all four questions, with section bookmarks and an experimental summary. Q2 occupies pages 7-12, exactly six pages including tables/figures. The page ranges and checks are in [combined_report_check.json](combined_report_check.json).

Rebuild from repository root:

```bash
python -m pip install -r report/requirements_report.txt
python report/build_combined_report.py
```

The builder uses the supplied Q1 notebook/results and the Q2/Q3/Q4 section PDFs. Hindi text is shaped with an embedded Devanagari font; Windows defaults to Nirmala, while other systems can pass `--font /path/to/NotoSansDevanagari.ttf`. It does not retrain models or fabricate measurements. See the combined root README for reproduction/submission instructions and [CONTRIBUTIONS.md](../CONTRIBUTIONS.md) for question ownership.
