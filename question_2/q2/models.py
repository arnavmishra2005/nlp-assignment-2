"""Explicit two-layer BiGRU attention baseline and custom RoBERTa pooling head."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, states, mask):
        scores = self.score(torch.tanh(self.projection(states))).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), states).squeeze(1), weights


class BiGRUAttention(nn.Module):
    def __init__(self, embeddings, config):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.gru = nn.GRU(embeddings.shape[1], config["hidden_dim"],
                          num_layers=config["num_layers"], batch_first=True,
                          bidirectional=True, dropout=config["dropout"])
        self.attention = AdditiveAttention(2 * config["hidden_dim"], config["attention_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(2 * config["hidden_dim"], 5)

    def forward(self, input_ids, lengths, return_attention=False):
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        states, _ = self.gru(packed)
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=input_ids.shape[1])
        mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
        pooled, weights = self.attention(states, mask)
        logits = self.classifier(self.dropout(pooled))
        return (logits, weights) if return_attention else logits


def masked_mean(states, attention_mask):
    # Includes <s> and </s>; excludes every padding token, as specified in the PDF.
    mask = attention_mask.unsqueeze(-1).to(states.dtype)
    return (states * mask).sum(1) / mask.sum(1).clamp_min(1)


class RobertaRating(nn.Module):
    def __init__(self, config, pooling="mean", pretrained_path=None, encoder_config=None):
        super().__init__()
        from transformers import RobertaConfig, RobertaModel
        if pooling not in ("mean", "first"):
            raise ValueError("Pooling must be mean or first")
        self.pooling = pooling
        if pretrained_path:
            self.encoder = RobertaModel.from_pretrained(pretrained_path, add_pooling_layer=False,
                                                       attn_implementation="sdpa", local_files_only=True)
        else:
            cfg = RobertaConfig.from_dict(encoder_config)
            cfg._attn_implementation = "sdpa"
            self.encoder = RobertaModel(cfg, add_pooling_layer=False)
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 5)

    def pool_and_classify(self, states, attention_mask):
        pooled = masked_mean(states, attention_mask) if self.pooling == "mean" else states[:, 0]
        return self.classifier(self.dropout(pooled))

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.pool_and_classify(output.last_hidden_state, attention_mask)


def make_model(model_name, config, prepared=None, metadata=None):
    if model_name == "baseline":
        import numpy as np
        if metadata is None:
            embeddings = torch.from_numpy(np.load(prepared / "embeddings.npy"))
        else:
            embeddings = torch.zeros(metadata["vocab_size"], config["baseline"]["embedding_dim"])
        return BiGRUAttention(embeddings, config["baseline"])
    return RobertaRating(config["roberta"], pooling="first" if model_name == "roberta_first" else "mean",
                         pretrained_path=prepared / "pretrained" if metadata is None else None,
                         encoder_config=metadata.get("encoder_config") if metadata else None)


def load_trained(directory, device):
    from pathlib import Path
    from .common import read_json
    directory = Path(directory)
    run = read_json(directory / "run.json")
    model = make_model(run["model"], run["config"], metadata=run["metadata"])
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["signature"] != run["signature"]:
        raise ValueError("Checkpoint signature mismatch")
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    return model.to(device).eval(), run

