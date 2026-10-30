"""Focused tests for masking, shifting, optimization, and cached inference."""
import unittest

import torch
from torch.nn import functional as F
from rouge_score import rouge_scorer
from unittest.mock import patch

from data import collate
from evaluate import sentences, summary_text
from model import Attention, PositionalEncoding, Transformer
from verify_inputs import rows_fingerprint


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = Transformer(24, width=16, heads=4, layers=2, hidden=32, dropout=0).eval()
        self.source = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]])
        self.target = torch.tensor([[2, 9, 10], [2, 11, 12]])

    def test_causal_mask_prevents_future_leakage(self):
        changed = self.target.clone()
        changed[:, 2] = 15
        torch.testing.assert_close(self.model(self.source, self.target)[:, :2],
                                   self.model(self.source, changed)[:, :2])

    def test_source_padding_is_ignored(self):
        padded = F.pad(self.source, (0, 3), value=0)
        torch.testing.assert_close(self.model(self.source, self.target),
                                   self.model(padded, self.target))

    def test_attention_masks_have_zero_mass(self):
        memory, mask = self.model.encode(self.source)
        _, weights = self.model.decode(self.target, memory, mask)
        self.assertTrue(torch.all(weights.masked_select(mask.expand_as(weights)) == 0))
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))

    def test_target_padding_is_masked(self):
        target = torch.tensor([[2, 0, 9]])
        # Changing the pad embedding cannot alter the non-pad final query:
        # source has no pads and the middle target key must be blocked.
        source = torch.tensor([[4, 5]])
        before = self.model(source, target)[:, -1].detach()
        with torch.no_grad():
            self.model.embedding.weight[0].fill_(100)
        torch.testing.assert_close(before, self.model(source, target)[:, -1])

    def test_incremental_generation_matches_full_prefix(self):
        cached, _ = self.model.generate(self.source, max_tokens=8)
        prefix = torch.full((2, 1), 2, dtype=torch.long)
        done = torch.zeros(2, dtype=torch.bool)
        memory, mask = self.model.encode(self.source)
        for step in range(cached.size(1)):
            logits = self.model.decode(prefix, memory, mask)[0][:, -1]
            logits[:, [0, 2]] = float('-inf')
            token = logits.argmax(-1).masked_fill(done, 3)
            torch.testing.assert_close(cached[:, step], token)
            prefix = torch.cat([prefix, token[:, None]], dim=1)
            done |= token.eq(3)

    def test_shift_and_padding(self):
        src, inp, target = collate([(torch.tensor([4, 5]), torch.tensor([2, 6, 3])),
                                    (torch.tensor([7]), torch.tensor([2, 3]))])
        self.assertEqual(inp.tolist(), [[2, 6], [2, 3]])
        self.assertEqual(target.tolist(), [[6, 3], [3, 0]])
        self.assertEqual(src.tolist(), [[4, 5], [7, 0]])

    def test_loss_ignores_padding(self):
        logits = torch.randn(2, 3, 24)
        labels = torch.tensor([[4, 3, 0], [7, 0, 0]])
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=0)
        expected = F.cross_entropy(logits[labels != 0], labels[labels != 0])
        torch.testing.assert_close(loss, expected)

    def test_optimizer_can_reduce_loss(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        labels = torch.tensor([[9, 10, 3], [11, 12, 3]])
        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            loss = F.cross_entropy(self.model(self.source, self.target).flatten(0, 1), labels.flatten())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0] / 2)

    def test_positional_encoding(self):
        values = PositionalEncoding(16)(torch.zeros(1, 3, 16))
        torch.testing.assert_close(values[0, 0, 0::2], torch.zeros(8))
        torch.testing.assert_close(values[0, 0, 1::2], torch.ones(8))
        self.assertFalse(torch.equal(values[:, 0], values[:, 1]))

    def test_sentence_newlines(self):
        self.assertEqual(sentences('First sentence. Second sentence!\nThird?'),
                         ['First sentence.', 'Second sentence!', 'Third?'])

    def test_generation_encodes_once_and_stops_at_eos(self):
        with torch.no_grad():
            self.model.output.weight.zero_()
            self.model.output.bias.zero_()
            self.model.output.bias[3] = 10
        with patch.object(self.model, 'encode', wraps=self.model.encode) as encode:
            generated, attention = self.model.generate(self.source, capture=True)
            self.assertEqual(encode.call_count, 1)
        self.assertEqual(generated.tolist(), [[3], [3]])
        self.assertEqual(tuple(attention.shape), (2, 1, 4))

    def test_rouge_perfect_reference(self):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                                         use_stemmer=True)
        text = 'The city council approved the plan.\nNew schools will open next year.'
        for score in scorer.score(text, text).values():
            self.assertAlmostEqual(score.fmeasure, 1.0)

    def test_data_fingerprint_detects_changed_text_and_order(self):
        rows = [dict(id='1', article='First article', highlights='First summary'),
                dict(id='2', article='Second article', highlights='Second summary')]
        before = rows_fingerprint(rows)
        self.assertNotEqual(before, rows_fingerprint(list(reversed(rows))))
        rows[0]['article'] = 'Wrong cached article'
        self.assertNotEqual(before, rows_fingerprint(rows))

    def test_data_fingerprint_rejects_duplicate_ids(self):
        row = dict(id='1', article='Article', highlights='Summary')
        with self.assertRaises(AssertionError):
            rows_fingerprint([row, row])


if __name__ == '__main__':
    unittest.main(verbosity=2)
