from typing import Literal

import torch
import torch.nn as nn


class SimpleRNNCell(nn.Module):
    def __init__(
        self, input_size: int, 
        hidden_size: int,
        init_distribution: Literal["normal", "uniform"] = "normal"
    ) -> None:
        super(SimpleRNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        init_method_ = nn.init.xavier_normal_ if init_distribution == "normal" else nn.init.xavier_uniform_
        W_ih = torch.empty((hidden_size, input_size))
        init_method_(W_ih)
        self.W_ih = nn.Parameter(W_ih)
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))

        W_hh = torch.empty(hidden_size, hidden_size)
        init_method_(W_hh)
        self.W_hh = nn.Parameter(W_hh)
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        input_feat = (self.W_ih @ x.t()).t() + self.b_ih # ()
        hidden_feat = (self.W_hh @ h_prev.t()).t() + self.b_hh

        h_t = torch.tanh(input_feat + hidden_feat)
        return h_t


class SimpleRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layer: int = 1,
    ) -> None:
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.rnn_cells = nn.ModuleList([
            SimpleRNNCell(
                input_size if i == 0 else hidden_size, hidden_size
            ) for i in range(num_layer)
        ])
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        B = input_seq.size(0)
        S = input_seq.size(1)

        # Initalize hidden state for each layer
        hidden_states = [
            torch.zeros(
                B, self.hidden_size, dtype=input_seq.dtype, device=input_seq.device
            ) for _ in range(self.num_layer)
        ]

        outputs = []
        for t in range(S):
            current_input = input_seq[:, t, :]
            for i in range(self.num_layer):
                if i == 0:
                    hidden_states[i] = self.rnn_cells[i](current_input, hidden_states[i])
                else:
                    hidden_states[i] = self.rnn_cells[i](hidden_states[i - 1], hidden_states[i])
            outputs.append(hidden_states[-1])
        
        return torch.stack(outputs, dim=1) # Stack outputs to get sequence output
    

class SimpClassificationRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        rnn_hidden_size: int,
        num_layer: int,
        num_class: int,
    ) -> None:
        super(SimpClassificationRNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, model_dim)
        self.rnn = SimpleRNN(model_dim, rnn_hidden_size, num_layer=num_layer)

        self.fc = nn.Linear(rnn_hidden_size, num_class)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(input_seq)
        x = self.rnn(embeds)
        logits = self.fc(x[:, -1, :])
        return logits
    

class TorchClassificationRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        rnn_hidden_size: int,
        num_layer: int,
        num_class: int,
    ) -> None:
        super(TorchClassificationRNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, model_dim)
        self.rnn = nn.RNN(
            input_size=model_dim,
            hidden_size=rnn_hidden_size,
            num_layers=num_layer,
            batch_first=True,
            nonlinearity="tanh",
            bias=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(rnn_hidden_size, num_class)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(input_seq)

        x, _ = self.rnn(embeds)
        logits = self.fc(x[:, -1, :])
        return logits
