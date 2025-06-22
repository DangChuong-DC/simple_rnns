import torch
import torch.nn as nn


class SimpleGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(SimpleGRUCell, self).__init__()
        self.hidden_size = hidden_size

        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        xh_combined = torch.cat([x, hx], dim=-1)

        r = torch.sigmoid(self.W_r(xh_combined)) # Reset gate
        z = torch.sigmoid(self.W_z(xh_combined)) # Update gate

        xrh_combined = torch.cat([x, r * hx], dim=-1)
        n = torch.tanh(self.W_h(xrh_combined))
        h = (1 - z) * n + z * hx
        return h
    

class SimpleGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int = 1):
        super(SimpleGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.gru = nn.ModuleList(
            [SimpleGRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layer)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.size()

        hidden_states = [
            torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layer)
        ]
        
        output = []
        for t in range(S):
            for i in range(self.num_layer):
                if i == 0:
                    hidden_states[i] = self.gru[i](x[:, t, :], hidden_states[i])
                else:
                    hidden_states[i] = self.gru[i](hidden_states[i - 1], hidden_states[i])
            output.append(hidden_states[-1])

        return torch.stack(output, dim=1)


class SimpleClassifierGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int, 
        hidden_size: int,
        num_layer: int = 1,
        num_class: int = 1,
        dropout_p: float = 0.1,
    ) -> None:
        super(SimpleClassifierGRU, self).__init__()
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(vocab_size, input_size)
        self.gru = SimpleGRU(input_size, hidden_size, num_layer=num_layer)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) # Shape: (B, S, input_size)
        gru_output = self.gru(x)  # Shape: (B, S, H)
        last_hidden_state = gru_output[:, -1, :]  # Take the last time step
        if self.dropout_p > 0:
            last_hidden_state = nn.functional.dropout(last_hidden_state, p=self.dropout_p, training=self.training)
        logits = self.fc(last_hidden_state)  # Shape: (B, output_size)
        return logits
