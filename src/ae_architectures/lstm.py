import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
    ) -> None:
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.decoder = LSTMDecoder(
            sequence_length, hidden_size, latent_size, input_size, num_layers
        )

    def forward(self, x: torch.Tensor):
        encoded_x = self.encoder(x)
        output = self.decoder(encoded_x)
        return output


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (_, _) = self.encoder_lstm(x)
        output = output.sum(dim=1)
        return output


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        hidden_size: int,
        latent_size: int,
        input_size: int,
        num_layers: int,
    ) -> None:
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.decoder = nn.LSTM(hidden_size, latent_size, num_layers, batch_first=True)
        self.decoder_fc = nn.Linear(latent_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        output, _ = self.decoder(output)
        output = self.decoder_fc(output)
        return output
