import torch
import torch.nn as nn

from src.utils.inference import reparameterization_trick


class LSTMVariationalAutoEncoder(nn.Module):
    """Variational AutoEncoder based on LSTM architecture"""

    def __init__(
        self,
        sequence_length: int,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
    ) -> None:
        super(LSTMVariationalAutoEncoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_size, num_layers)
        self.decoder = LSTMDecoder(
            sequence_length, latent_size, hidden_size, input_size, num_layers
        )

    def forward(self, x: torch.Tensor):
        encoded_x, mean, log_var = self.encoder(x)
        output = self.decoder(encoded_x)
        return output, mean, log_var


class LSTMEncoder(nn.Module):
    """Encoder based on LSTM architecture"""

    def __init__(
        self, input_size: int, hidden_size: int, latent_size: int, num_layers: int
    ) -> None:
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.mlp = nn.Linear(hidden_size, latent_size)
        self.distribution_mean = nn.Linear(latent_size, latent_size)
        self.distribution_var = nn.Linear(latent_size, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (_, _) = self.encoder_lstm(x)
        x = x.sum(dim=1)
        x = self.mlp(x)

        mean = self.distribution_mean(x)
        log_var = self.distribution_var(x)

        z = reparameterization_trick(mean, log_var)
        return x, mean, log_var


class LSTMDecoder(nn.Module):
    """Decoder based on LSTM architecture"""

    def __init__(
        self,
        sequence_length: int,
        latent_size: int,
        hidden_size: int,
        input_size: int,
        num_layers: int,
    ) -> None:
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length

        self.mlp1 = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.mlp2 = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x = self.mlp1(x)
        x, _ = self.decoder(x)
        x = self.mlp2(x)
        return x
