"""A post-normalization Transformer written using elementary PyTorch layers."""
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, width, limit=512):
        super().__init__()
        position = torch.arange(limit).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        table = torch.zeros(limit, width)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer('table', table.unsqueeze(0))

    def forward(self, x, offset=0):
        return x + self.table[:, offset:offset + x.size(1)]


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        return x.view(x.size(0), x.size(1), self.heads, self.head_width).transpose(1, 2)

    def project_kv(self, x):
        return self.split(self.key(x)), self.split(self.value(x))

    def attend(self, query, key, value, blocked=None):
        scores = self.split(self.query(query)) @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_width)
        if blocked is not None:
            scores = scores.masked_fill(blocked, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        context = self.dropout(weights) @ value
        context = context.transpose(1, 2).contiguous().flatten(2)
        return self.output(context), weights

    def forward(self, query, memory, blocked=None):
        key, value = self.project_kv(memory)
        return self.attend(query, key, value, blocked)


class FeedForward(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, width))

    def forward(self, x):
        return self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.attention = Attention(width, heads, dropout)
        self.feed_forward = FeedForward(width, hidden, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, source_mask):
        attended, _ = self.attention(x, x, source_mask)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.self_attention = Attention(width, heads, dropout)
        self.cross_attention = Attention(width, heads, dropout)
        self.feed_forward = FeedForward(width, hidden, dropout)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(width) for _ in range(3)]
        self.dropout = nn.Dropout(dropout)

    def finish(self, x, memory_kv, source_mask):
        attended, weights = self.cross_attention.attend(x, *memory_kv, source_mask)
        x = self.norm2(x + self.dropout(attended))
        return self.norm3(x + self.dropout(self.feed_forward(x))), weights

    def forward(self, x, memory, target_mask, source_mask):
        attended, _ = self.self_attention(x, x, target_mask)
        x = self.norm1(x + self.dropout(attended))
        return self.finish(x, self.cross_attention.project_kv(memory), source_mask)

    def step(self, x, memory_kv, source_mask, cache):
        # Cache projected keys/values so greedy generation processes one new token.
        key, value = self.self_attention.project_kv(x)
        if cache is not None:
            key = torch.cat([cache[0], key], dim=2)
            value = torch.cat([cache[1], value], dim=2)
        attended, _ = self.self_attention.attend(x, key, value)
        x = self.norm1(x + self.dropout(attended))
        x, weights = self.finish(x, memory_kv, source_mask)
        return x, weights, (key, value)


class Transformer(nn.Module):
    def __init__(self, vocab_size, width=64, heads=4, layers=2, hidden=256,
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.width, self.pad_id = width, pad_id
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.position = PositionalEncoding(width)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([EncoderLayer(width, heads, hidden, dropout)
                                      for _ in range(layers)])
        self.decoder = nn.ModuleList([DecoderLayer(width, heads, hidden, dropout)
                                      for _ in range(layers)])
        self.output = nn.Linear(width, vocab_size)

    def embed(self, ids, offset=0):
        return self.dropout(self.position(self.embedding(ids) * math.sqrt(self.width), offset))

    def encode(self, source):
        mask = source.eq(self.pad_id)[:, None, None, :]
        x = self.embed(source)
        for layer in self.encoder:
            x = layer(x, mask)
        return x, mask

    def decode(self, target, memory, source_mask):
        length = target.size(1)
        causal = torch.ones(length, length, dtype=torch.bool, device=target.device).triu(1)
        mask = causal[None, None] | target.eq(self.pad_id)[:, None, None, :]
        x = self.embed(target)
        for layer in self.decoder:
            x, weights = layer(x, memory, mask, source_mask)
        return self.output(x), weights

    def forward(self, source, target):
        memory, mask = self.encode(source)
        return self.decode(target, memory, mask)[0]

    @torch.inference_mode()
    def generate(self, source, bos_id=2, eos_id=3, max_tokens=128, capture=False):
        """Greedy generation; attention row t predicts generated token t."""
        self.eval()
        memory, source_mask = self.encode(source)  # Exactly once per article.
        memory_kv = [layer.cross_attention.project_kv(memory) for layer in self.decoder]
        caches = [None] * len(self.decoder)
        token = torch.full((source.size(0), 1), bos_id, device=source.device, dtype=torch.long)
        done = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        outputs, attention = [], []
        for position in range(max_tokens):
            x = self.embed(token, position)
            for index, layer in enumerate(self.decoder):
                x, weights, caches[index] = layer.step(x, memory_kv[index], source_mask, caches[index])
            logits = self.output(x[:, -1])
            logits[:, [self.pad_id, bos_id]] = float('-inf')
            token = logits.argmax(-1).masked_fill(done, eos_id).unsqueeze(1)
            outputs.append(token)
            if capture:
                attention.append(weights.mean(1)[:, 0].cpu())
            done |= token[:, 0].eq(eos_id)
            if done.all():
                break
        ids = torch.cat(outputs, dim=1)
        return ids, torch.stack(attention, dim=1) if capture else None
