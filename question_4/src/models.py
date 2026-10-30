from torch import nn


class StoryLM(nn.Module):
    def __init__(self, vocab_size, rnn_type, embed_dim, hidden_dim, num_layers, dropout, pad_id):
        super().__init__()
        rnn_class = {"rnn": nn.RNN, "lstm": nn.LSTM}[rnn_type]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = rnn_class(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        embedded = self.dropout(self.embedding(inputs))
        outputs, hidden = self.rnn(embedded, hidden)
        logits = self.output(self.dropout(outputs))
        return logits, hidden
