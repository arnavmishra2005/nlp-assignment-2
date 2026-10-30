import torch
import torch.nn as nn
import torch.nn.functional as F
from data import PAD, SOS, EOS, UNK


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256, drop=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.drop = nn.Dropout(drop)

    def forward(self, src, src_len):
        x = self.drop(self.embed(src))
        packed = nn.utils.rnn.pack_padded_sequence(x, src_len, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        return out, (h[0], c[0])        # out = h1..hn, (h, c) = final state


class Attention(nn.Module):
    # score(s, h_i) = v . tanh(W_s s + W_h h_i)
    def __init__(self, hid):
        super().__init__()
        self.W_h = nn.Linear(hid, hid, bias=False)
        self.W_s = nn.Linear(hid, hid)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, s_prev, enc_out, enc_proj, mask):
        scores = self.v(torch.tanh(enc_proj + self.W_s(s_prev).unsqueeze(1))).squeeze(2)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=1)
        context = (alpha.unsqueeze(2) * enc_out).sum(dim=1)
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256, drop=0.2, use_attention=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.attn = Attention(hid) if use_attention else None
        self.cell = nn.LSTMCell(emb + (hid if use_attention else 0), hid)
        self.out = nn.Linear(hid, vocab_size)
        self.drop = nn.Dropout(drop)

    def step(self, prev_word, state, enc_out, enc_proj, mask):
        x = self.drop(self.embed(prev_word))
        alpha = None
        if self.attn is not None:
            context, alpha = self.attn(state[0], enc_out, enc_proj, mask)
            x = torch.cat([x, context], dim=1)
        h, c = self.cell(x, state)
        return h, (h, c), alpha


class Translator(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=128, hid=256, drop=0.3, use_attention=False):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb, hid, drop)
        self.decoder = Decoder(tgt_vocab, emb, hid, drop, use_attention)

    def start(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        enc_proj = self.decoder.attn.W_h(enc_out) if self.decoder.attn else None
        return enc_out, enc_proj, state, src != PAD

    def forward(self, src, src_len, tgt_in):
        enc_out, enc_proj, state, mask = self.start(src, src_len)
        hidden = []
        for t in range(tgt_in.size(1)):
            h, state, _ = self.decoder.step(tgt_in[:, t], state, enc_out, enc_proj, mask)
            hidden.append(h)
        return self.decoder.out(torch.stack(hidden, dim=1))

    @torch.no_grad()
    def translate(self, src, src_len, max_len=40, ban_unk=True):
        enc_out, enc_proj, state, mask = self.start(src, src_len)
        word = torch.full((src.size(0),), SOS, dtype=torch.long)
        words, alphas = [], []
        for _ in range(max_len):
            h, state, alpha = self.decoder.step(word, state, enc_out, enc_proj, mask)
            logits = self.decoder.out(h)
            logits[:, [PAD, SOS] + ([UNK] if ban_unk else [])] = -1e9
            word = logits.argmax(dim=1)
            words.append(word)
            if alpha is not None:
                alphas.append(alpha)
            if (torch.stack(words).eq(EOS).any(dim=0)).all():
                break
        alphas = torch.stack(alphas, dim=1) if alphas else None
        return torch.stack(words, dim=1), alphas
