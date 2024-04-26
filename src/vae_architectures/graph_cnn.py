import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphVariationalAutoEncoder(nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = GraphEncoder(num_features, hidden_channels, num_layers)
        self.decoder = GraphDecoder(hidden_channels, num_features)

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(x)
        return x

class GraphEncoder(nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        return x

class GraphDecoder(nn.Module):
    def __init__(self, hidden_channels, num_features):
        super(GraphDecoder, self).__init__()
        self.fc = nn.Linear(hidden_channels, num_features)

    def forward(self, x):
        x = self.fc(x)
        return x

class GraphAutoEncoder(nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = GraphEncoder(num_features, hidden_channels, num_layers)
        self.decoder = GraphDecoder(hidden_channels, num_features)

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(x)
        return x
