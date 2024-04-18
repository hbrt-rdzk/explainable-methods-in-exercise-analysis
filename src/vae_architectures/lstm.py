import torch
import torch.nn as nn


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
        self.distribution_mean = nn.Linear(hidden_size, latent_size)
        self.distribution_var = nn.Linear(hidden_size, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (_, _) = self.encoder_lstm(x)
        output = output.sum(dim=1)

        mean = self.distribution_mean(output)
        log_var = self.distribution_var(output)

        z = self.reparameterization_trick(mean, log_var)
        return output, mean, log_var

    def reparameterization_trick(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z


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
        self.decoder = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        output, _ = self.decoder(output)
        output = self.decoder_fc(output)
        return output
