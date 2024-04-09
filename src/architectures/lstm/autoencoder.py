import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, latent_size: int, num_layers: int
    ) -> None:
        super(LSTMAutoEncoder, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.hidden_size = hidden_size
        # encoder layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, hidden_size)

        # decoder layers
        self.decoder = nn.LSTM(hidden_size, latent_size, num_layers, batch_first=True)
        self.decoder_fc = nn.Linear(latent_size, input_size)

    def forward(self, x: torch.Tensor, lengths: list[int] = None):
        if lengths is None:
            lengths = [len(seq) for seq in x]

        # encoder
        output, (_, _) = self.encoder(x)
        output = output.sum(dim=1)
        z = self.encoder_fc(output)
        z = self.dropout(z)

        # decoder
        output = output.unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder(output)
        output = self.decoder_fc(output)
        return output
