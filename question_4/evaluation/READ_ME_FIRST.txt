BLINDED HUMAN EVALUATION — PLOT-TWIST GENERATOR

Purpose
Collect genuine human ratings of the 30 anonymised generated continuations (S01–S30).
The rating sheets contain the exact sample text and blank rating fields. No scores have
been pre-filled. AI-simulated scores are kept separate and must not be entered as human data.

Minimum collection plan
- Obtain ratings for at least 10 distinct samples.
- Aim for three independent human evaluators per sample wherever feasible.
- This packet contains all 30 samples in three separate evaluator sheets. Ideally, each
  evaluator rates all 30; at minimum, coordinate coverage so at least 10 samples have
  three independent ratings.
- Each evaluator should work independently and must not see another evaluator's scores.
- Do not reveal or infer the generating model or decoding strategy. Do not discuss ratings
  until all sheets are submitted.

Scale (whole numbers only)
1 = Very poor; 2 = Poor; 3 = Adequate; 4 = Good; 5 = Excellent.

Criteria
- Coherence: logical connection and consistency of events/references.
- Grammar: sentence structure, word choice, punctuation and readability.
- Prompt relevance: meaningfully develops the supplied opening premise.
- Creativity/surprise: original or unexpected development that still makes sense.
Judge the generated continuation, not the supplied opening. Do not reward length, randomness,
or bizarre wording by itself.

Files
- Human_Rater_1_Rating_Sheet_Blank.csv, Human_Rater_2_Rating_Sheet_Blank.csv, Human_Rater_3_Rating_Sheet_Blank.csv:
  independent forms; all score cells, rater_id and notes are blank for the evaluator to fill.
- Human_All_Ratings_Blank_Template.csv: consolidated 90-row template for data entry after
  independent forms are returned.
- ../results/human_eval_blinded.csv: project master copy, also blank.

After collection
Enter each evaluator's actual ratings without changing sample IDs or story text. Record a
non-identifying evaluator code (e.g. H1/H2/H3) in rater_id. Calculate criterion means,
per-sample ranges/standard deviations, and compare sample-level human ratings with
sample-level model metrics only if aligned per-sample perplexity data is available.
Never describe AI-simulated ratings as human ratings.
