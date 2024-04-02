import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, num_classes: int
    ) -> None:
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, lengths: list[int]):
        packed_sequence = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden_state, _) = self.lstm(packed_sequence)
        output = hidden_state[-1, :, :]
        output = self.fc(output)
        output = self.softmax(output)
        return output
