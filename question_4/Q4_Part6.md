# Part 6 — Final Analysis

Numbers come from our own result files. Only the generated text is measured (prompt removed). Samples are small, so results are suggestive, not proof.

| Model | Test perplexity | Greedy repetition | Greedy loops | Temp 0.8 Distinct-2 | Top-k Distinct-2 |
|---|---|---|---|---|---|
| RNN | 16.88 | 0.363 | 4 of 7 | 0.754 | 0.723 |
| LSTM | 13.40 | 0.211 | 2 of 7 | 0.771 | 0.710 |

| LSTM temperature | Distinct-2 | Repetition |
|---|---|---|
| 0.5 | 0.669 | 0.067 |
| 0.8 | 0.824 | 0.031 |
| 1.2 | 0.962 | 0.004 |

**1. How do RNN and LSTM differ?**
The LSTM predicts text better: test perplexity 13.40 vs 16.88. Its gates help it keep information longer. It costs about 21% more parameters and 38% more training time. Both were still improving at epoch 10.

**2. Which is more coherent and less repetitive?**
The LSTM is less repetitive, mostly under greedy decoding (0.211 vs 0.363). Both ignore the unusual prompts and write generic stories. *TODO: add the human coherence scores once collected.*

**3. How does temperature affect creativity and grammar?**
Higher temperature gives more varied words but worse grammar. At 0.5 the text is readable but samey. At 0.8 it is a good balance. At 1.2 it falls apart (e.g. "it waited what was jane standing in the blue - `<unk>` sparkling coming water").

**4. Do high diversity scores guarantee better stories?**
No. Temperature 1.2 has the highest diversity (Distinct-2 ≈ 0.96) and the worst text. Diversity counts unique words, not whether they make sense.

**5. Does lower perplexity mean better storytelling?**
Only a little. The LSTM loops less and reads more smoothly, but neither model follows the prompt or makes a real twist. *TODO: compare with the human ratings (say if they are AI-simulated).*

**6. What failures are most common, and how to reduce them?**
- Ignoring the prompt → more data/training.
- Repetition loops (greedy: RNN 4/7, LSTM 2/7) → repetition penalty or sampling.
- Character drift → bigger model, longer training.
- Gibberish at high temperature → use 0.8 or top-k.
- Broken characters (`€ ™`) and `<unk>` → fix text cleaning and retrain.

These fixes were not tested.

**7. Limitations**
- Small data: 20,000 stories, word-level vocabulary (6,695 words).
- Undertrained: 10 epochs, one seed (42).
- Stories cut at 200 tokens; outputs cut at 80.
- No planning, so no real plot twists.
- Small evaluation: 7 prompts, few samples, small human study.
