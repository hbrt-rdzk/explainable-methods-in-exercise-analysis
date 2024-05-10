import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from src.utils.inference import reparameterization_trick


class GraphVariationalAutoEncoder(nn.Module):
    def __init__(
        self, input_size: int, num_nodes: int, hidden_size: int, latent_size: int
    ) -> None:
        super(GraphVariationalAutoEncoder, self).__init__()
        self.encoder = GraphEncoder(num_nodes, input_size, hidden_size, latent_size)
        self.decoder = GraphDecoder(num_nodes, latent_size, hidden_size, input_size)

    def forward(self, x):
        x, mean, log_var = self.encoder(x)
        x = self.decoder(x)
        return x, mean, log_var


class GraphEncoder(nn.Module):
    def __init__(
        self, num_nodes: int, input_size: int, hidden_size: int, latent_size: int
    ) -> None:
        super(GraphEncoder, self).__init__()

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.conv1 = GraphConvolution(input_size, hidden_size, node_n=num_nodes)
        self.conv2 = GraphConvolution(hidden_size, hidden_size, node_n=num_nodes)

        self.mlp = nn.Linear(hidden_size, latent_size)
        self.distribution_mean = nn.Linear(latent_size, latent_size)
        self.distribution_var = nn.Linear(latent_size, latent_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = torch.sum(x, dim=1)

        x = self.mlp(x)
        mean = self.distribution_mean(x)
        log_var = self.distribution_var(x)

        z = reparameterization_trick(mean, log_var)
        return x, mean, log_var


class GraphDecoder(nn.Module):
    def __init__(
        self, num_nodes, latent_size: int, hidden_size: int, input_size: int
    ) -> None:
        super(GraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.relu = nn.ReLU()

        self.mlp1 = nn.Linear(latent_size, hidden_size)
        self.conv1 = GraphConvolution(hidden_size, hidden_size, node_n=num_nodes)
        self.conv2 = GraphConvolution(hidden_size, hidden_size, node_n=num_nodes)
        self.mlp2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.num_nodes, 1)
        x = self.mlp1(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.mlp2(x)
        x = x.permute(0, 2, 1)
        return x


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=45):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
