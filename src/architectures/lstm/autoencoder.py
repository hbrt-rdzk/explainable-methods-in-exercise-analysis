import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class LSTMAutoEncoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, latent_dim: int, num_layers: int
    ) -> None:
        super(LSTMAutoEncoder, self).__init__()
        # encoder layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)

        # decoder layers
        self.decoder_fc = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor, lengths: list[int] = None):
        if lengths is None:
            lengths = [len(seq) for seq in x]
        encoder_output, (hidden_states, _) = self.encoder(x)
        hidden_states = hidden_states.transpose(1, 0)
        output = hidden_states[:, -1, :]
        z = self.encoder_fc(output)

        # decoder
        output = self.decoder_fc(z)
        output = torch.stack(
            [
                embeddings.unsqueeze(0).repeat_interleave(lengths, dim=0)
                for embeddings, lengths in zip(output, lengths)
            ],
            dim=0,
        )
        output, _ = self.decoder(output)
        return output
