# Human evaluation instructions

The assignment requires genuine ratings; this package does not fabricate them.

1. Give evaluators `human_eval_blinded.csv` or `human_eval_3_rater_template.csv` only. Do **not** share `human_eval_key.csv` with evaluators until ratings are complete.
2. For each sample, rate the continuation against its prompt from 1 (very poor) to 5 (excellent) on coherence, grammatical quality, prompt relevance, and creativity/surprise.
3. The three-rater template has one row per sample per evaluator (`R1`, `R2`, `R3`). Replace those labels with anonymized evaluator IDs if preferred. Each evaluator should score independently.
4. Keep the sample IDs unchanged. After collection, calculate the mean for each criterion and report the number of actual evaluators. Discuss disagreements and compare ratings with perplexity/automatic metrics.
5. Only after collection, use `human_eval_key.csv` to reveal which model and decoding strategy produced each sample. Do not describe blank forms as completed evaluation.

The selected set contains 30 continuations: 5 samples from each of the six model/strategy combinations (RNN/LSTM × greedy/temperature/top-k). It includes the prompt separately and preserves the continuation text from the saved challenge outputs without rewriting it.
