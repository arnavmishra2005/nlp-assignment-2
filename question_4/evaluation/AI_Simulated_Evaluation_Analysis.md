# Plot-Twist Generator — Evaluation Analysis

## Evaluation dataset and provenance

Thirty blinded samples (S01–S30) were rated in three AI-simulated passes, producing 90 records. The three passes are rubric perspectives from one AI-assisted workflow, not independent human evaluators. The ratings are used as a proxy dataset for the requested descriptive evaluation; they must not be represented as genuine human ratings.

## Criterion means across all 90 ratings

| Criterion | Mean (1–5) |
|---|---:|
| Coherence | 1.83 |
| Grammar | 2.98 |
| Prompt relevance | 1.00 |
| Creativity/surprise | 1.40 |
| **Overall mean** | **1.80** |

## Means by AI-simulated rating pass

| Pass | Coherence | Grammar | Relevance | Creativity/surprise |
|---|---:|---:|---:|---:|
| AI-Simulated-R1 | 1.83 | 2.67 | 1.00 | 1.07 |
| AI-Simulated-R2 | 1.83 | 3.60 | 1.00 | 1.07 |
| AI-Simulated-R3 | 1.83 | 2.67 | 1.00 | 2.07 |

## Disagreement analysis

Disagreement is reported as the range across the three passes for each sample and criterion. This is descriptive variation between AI rubric passes, not human inter-rater reliability.

| Criterion | Samples with any disagreement | Maximum range | Exact three-way agreement |
|---|---:|---:|---:|
| Coherence | 0/30 | 0 | 100.0% |
| Grammar | 28/30 | 1 | 6.7% |
| Prompt relevance | 0/30 | 0 | 100.0% |
| Creativity/surprise | 30/30 | 1 | 0.0% |

Grammar differed in 28/30 samples, while creativity/surprise differed in all 30; each observed range was at most one point. Coherence and relevance were identical across the three passes for every sample. The prompt-fit/novelty pass assigned a higher creativity/surprise mean (2.07) than the other two passes (1.07 each).

## Qualitative synthesis

- Prompt drift was the dominant weakness: continuations often abandoned the opening premise.
- Local continuity was weak, with unrelated characters, objects, and settings introduced without explanation.
- Grammar/readability was stronger than narrative quality, although awkward dialogue, punctuation artifacts, and unclear references appeared.
- Repetition often replaced plot progression and meaningful surprise.
- S14 and S22 were identified as locally more understandable, but neither meaningfully resolved its opening premise.

## Perplexity comparison

Archived test perplexity is 16.88 for the Vanilla RNN and 13.40 for the LSTM (lower is better). The proxy rating averages indicate weak narrative quality overall, particularly prompt relevance (1.00/5), despite the LSTM's lower perplexity. This is a qualitative juxtaposition only: sample-level perplexities and a verified mapping from S01–S30 to model are unavailable. No correlation coefficient or model-specific rating comparison is justified. Lower perplexity measures next-token prediction, not story relevance or plot quality.

## Blinding and provenance

No RNN/LSTM or decoding-strategy labels were inferred for S01–S30. These are AI-simulated proxy ratings, not genuine human ratings. They should not be described as three independent human evaluators or as evidence of human inter-rater reliability.