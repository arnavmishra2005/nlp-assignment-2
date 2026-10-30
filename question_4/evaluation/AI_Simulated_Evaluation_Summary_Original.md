# Plot Twist Generator — AI-Simulated Evaluation

## Important disclosure
This is an **AI-simulated assessment**, not a human-subject evaluation. The three evaluator columns represent three rubric perspectives produced in one AI-assisted review workflow. They are **not three independent human raters**, and the results must not be reported as human ratings or as evidence of inter-human reliability.

## Material evaluated
- Samples: 30 generated continuations (S01–S30)
- Ratings: 90 total rows (30 samples × 3 AI-simulated evaluator perspectives)
- Scale: 1 (very poor) to 5 (excellent)
- Dimensions: coherence, grammar, relevance to the given prompt, and creativity/surprise
- Text basis: the supplied `Plot_Twist_Generator_Blinded_Rating_Sheet.csv`

## Rubric
- **Coherence:** Do the events and references form a logically connected sequence?
- **Grammar:** Are sentence structure, word choice, punctuation, and readability acceptable?
- **Prompt relevance:** Does the continuation actually develop the supplied opening premise?
- **Creativity/surprise:** Does it introduce an original development or an effective unexpected turn?

## Evaluator perspectives
- **AI-Simulated-R1 — Strict narrative lens:** emphasizes causal continuity, unresolved references, and whether the story develops its premise.
- **AI-Simulated-R2 — Readability lens:** gives slightly more weight to surface readability and grammar while keeping plot relevance separate.
- **AI-Simulated-R3 — Prompt-fit/novelty lens:** emphasizes whether the opening premise is used and whether the continuation earns a surprising payoff.

These perspectives are deliberately identified as AI-simulated. Their differences are rubric-based, not independent human judgments.

## Aggregate results
| Dimension | Mean score (1–5) |
|---|---:|
| Coherence | 1.83 |
| Grammar | 2.98 |
| Prompt relevance | 1.00 |
| Creativity/surprise | 1.40 |
| **Overall mean across four dimensions** | **1.80** |

## Mean scores by evaluator perspective
| Evaluator | Coherence | Grammar | Relevance | Creativity/surprise |
| AI-Simulated-R1 | 1.83 | 2.67 | 1.00 | 1.07 |
| AI-Simulated-R2 | 1.83 | 3.60 | 1.00 | 1.07 |
| AI-Simulated-R3 | 1.83 | 2.67 | 1.00 | 2.07 |

## Exact agreement across the three AI-simulated perspectives
- coherence_1_5: 100.0% of samples received the same score from all three perspectives.
- grammar_1_5: 6.7% of samples received the same score from all three perspectives.
- relevance_1_5: 100.0% of samples received the same score from all three perspectives.
- creativity_surprise_1_5: 0.0% of samples received the same score from all three perspectives.

## Main qualitative findings
1. **Prompt drift is the dominant weakness.** Most continuations quickly abandon the supplied premise (for example, the detective, future letter, exam paper, village memory, or approaching moon).
2. **Local sentence-to-sentence continuity is often weak.** Stories frequently jump between unrelated people, objects, and settings without explaining their relationship.
3. **Surface grammar is generally stronger than narrative quality**, although malformed dialogue, punctuation artifacts, awkward phrasing, and unclear pronouns appear in several samples.
4. **Repetition limits development and surprise.** Repeated phrases such as being excited, thanking a bird, or climbing a tree often replace meaningful plot progression.
5. **A few samples have a locally understandable scene** (for example, S14's tree-climbing sequence and S22's short dialogue), but even these do not meaningfully resolve their opening premise.

## Interpretation and limitations
- The scores describe the 30 supplied samples only; they do not establish general model performance.
- This evaluation does not compare RNN against LSTM unless the sample-to-model mapping is supplied and verified.
- The evaluator perspectives are not independent people. Do not calculate or claim human inter-rater reliability from these scores.
- Scores are based on the story text available in the supplied CSV. They should be treated as a structured AI-assisted qualitative review, not ground-truth labels.
- For a formal assignment requiring human evaluation, replace or supplement this section with ratings collected independently from actual human evaluators using the blinded rating sheet.
