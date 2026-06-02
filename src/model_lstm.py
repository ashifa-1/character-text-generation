import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x, hidden):

        embeddings = self.embedding(x)

        lstm_out, hidden = self.lstm(
            embeddings,
            hidden
        )

        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data

        hidden = (
            weight.new_zeros(
                self.n_layers,
                batch_size,
                self.hidden_dim
            ),
            weight.new_zeros(
                self.n_layers,
                batch_size,
                self.hidden_dim
            )
        )

        return hidden