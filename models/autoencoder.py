import torch
from torch.nn import ReLU
from torch_geometric.nn import Sequential, GCNConv


class BaseGNNAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(BaseGNNAutoEncoder, self).__init__()
        torch.manual_seed(12345)

        self.encoder = Sequential('x, edge_index', [])
        self.decoder = Sequential('x, edge_index', [])

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        return x


class DeepGNNAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, middle_channels=128, bottleneck_channels=64):
        super(DeepGNNAutoEncoder, self).__init__()
        torch.manual_seed(12345)

        self.encoder = Sequential('x, edge_index', [
            (GCNConv(in_channels, middle_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(middle_channels, bottleneck_channels), 'x, edge_index -> x'),
        ])
        self.decoder = Sequential('x, edge_index', [
            (GCNConv(bottleneck_channels, middle_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(middle_channels, in_channels), 'x, edge_index -> x'),
        ])

    def __repr__(self) -> str:
        return 'DeepGNNAutoEncoder'


class ShallowGNNAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, bottleneck_channels=64):
        super(ShallowGNNAutoEncoder, self).__init__()
        torch.manual_seed(12345)

        self.encoder = Sequential('x, edge_index', [
            (GCNConv(in_channels, bottleneck_channels), 'x, edge_index -> x'),
        ])
        self.decoder = Sequential('x, edge_index', [
            (GCNConv(bottleneck_channels, in_channels), 'x, edge_index -> x'),
        ])

    def __repr__(self) -> str:
        return 'ShallowGNNAutoEncoder'
