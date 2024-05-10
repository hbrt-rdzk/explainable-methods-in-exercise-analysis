import torch
import torch.nn as nn

from src.utils.inference import reparameterization_trick


class SignalCNNVariationalAutoEncoder(nn.Module):
    """Variational AutoEncoder based on CNN architecture"""

    def __init__(
        self,
        sequence_length: int,
        input_size: int,
        hidden_size: int,
        latent_size: int,
    ) -> None:
        super(SignalCNNVariationalAutoEncoder, self).__init__()
        self.encoder = SignalCNNEncoder(
            sequence_length, input_size, hidden_size, latent_size
        )
        self.decoder = SignalCNNDecoder(
            sequence_length, latent_size, hidden_size, input_size
        )

    def forward(self, x: torch.Tensor):
        encoded_x, mean, log_var = self.encoder(x)
        output = self.decoder(encoded_x)
        return output, mean, log_var


class SignalCNNEncoder(nn.Module):
    """Encoder based on CNN architecture"""

    def __init__(
        self, sequence_length: int, input_size: int, hidden_size: int, latent_size: int
    ) -> None:
        super(SignalCNNEncoder, self).__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.mlp = nn.Linear(hidden_size, latent_size)
        self.distribution_mean = nn.Linear(latent_size, latent_size)
        self.distribution_var = nn.Linear(latent_size, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.sum(x, dim=2)

        x = self.mlp(x)
        mean = self.distribution_mean(x)
        log_var = self.distribution_var(x)

        z = reparameterization_trick(mean, log_var)
        return x, mean, log_var


class SignalCNNDecoder(nn.Module):
    """Decoder based on CNN architecture"""

    def __init__(
        self,
        sequence_length: int,
        latent_size: int,
        hidden_size: int,
        input_size: int,
    ) -> None:
        super(SignalCNNDecoder, self).__init__()
        self.before_flatten_length = sequence_length // 4
        self.relu = nn.ReLU()

        self.mlp1 = nn.Linear(latent_size, hidden_size * self.before_flatten_length)
        self.conv1 = nn.ConvTranspose1d(
            hidden_size, hidden_size, kernel_size=2, stride=2
        )
        self.conv2 = nn.ConvTranspose1d(
            hidden_size, hidden_size, kernel_size=3, stride=2
        )
        self.mlp2 = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1, self.before_flatten_length)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = x.permute(0, 2, 1)

        x = self.mlp2(x)
        return x
