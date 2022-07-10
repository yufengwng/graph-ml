"""Operators related to GCN."""

import torch
from torch import nn

from .typing import Optional, Tensor


class GCN(nn.Module):
    """ A graph convolutional network (GCN).

    Based on original GCN paper (Kipf & Welling, 2017). Activation is applied
    between layers but no activation is applied at last layer.

    ----------------------------------------------------------------------
    """

    def __init__(self, num_layers: int, in_dim: int, out_dim: int,
                 hidden_dim=16, bias=False, activation=nn.ReLU(),
                 dropout: Optional[float] = None):
        """
        # Parameters

        * `num_layers` - Number of convolution layers
        * `in_dim` - Dimension of input features
        * `out_dim` - Dimension of output features
        * `hidden_dim` - Dimension of hidden layers
        * `bias` - Whether to apply bias
        * `activation` - The activation function to apply between layers
        * `dropout` - Dropout probability
        """
        super().__init__()
        assert (num_layers > 0)

        self.n_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        if num_layers == 1:
            self.layers = nn.ModuleList([
                AdjConv(in_dim, out_dim, bias, activation=None)])
        else:
            layers = [AdjConv(in_dim, hidden_dim, bias, activation=activation)]
            for _ in range(num_layers - 2):
                layers.append(AdjConv(hidden_dim, hidden_dim, bias, activation=activation))
            layers.append(AdjConv(hidden_dim, out_dim, bias, activation=None))
            self.layers = nn.ModuleList(layers)

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, adj: Tensor, feats: Tensor) -> Tensor:
        """
        # Parameters

        * `adj` - Renormalized adjacency matrix of [n x n]
        * `feats` - Node input feature tensor of [n x in_dim]

        # Returns

        Node embeddings tensor of [n x out_dim]
        """
        embed = feats
        for layer in self.layers:
            embed = layer(adj, embed)
            if self.dropout is not None:
                embed = self.dropout(embed)
        return embed


class AdjConv(nn.Module):
    """An operator that performs graph convolution on adjacent neighbors.

    $$ H = \sigma(\hat{A} \cdot H \cdot W) $$

    Activation function is optional and will be applied after convolution.

    ----------------------------------------------------------------------
    """

    def __init__(self, in_dim: int, out_dim: int,
                 bias=False, activation: Optional[nn.Module] = None):
        """
        # Parameters

        * `in_dim` - Dimension of input features
        * `out_dim` - Dimension of output features
        * `bias` - Whether to apply bias
        * `activation` - The activation function to apply
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = activation

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, adj: Tensor, input: Tensor) -> Tensor:
        """
        # Parameters

        * `adj` - Renormalized adjacency matrix of [n x n]
        * `input` - Input tensor of [n x in_dim]

        # Returns

        Convolved output tensor of [n x out_dim]
        """
        output = torch.matmul(adj, self.linear(input))
        if self.act is not None:
            output = self.act(output)
        return output


def renormalize_adjacency(adj: Tensor) -> Tensor:
    """Returns renormalized adjacency matrix.

    $$ \hat{A} = D^{-1/2} \cdot (A + I_N) \cdot D^{-1/2} $$

    # Parameters

    * `adj` - Adjacency matrix where an edge is indicated by a 1 entry
    """
    n_nodes = adj.shape[0]
    assert (n_nodes == adj.shape[1])

    adj_eye = adj + torch.eye(n_nodes)
    deg_inv = torch.diag(adj_eye.sum(dim=1)).inverse().sqrt()
    adj_hat = torch.matmul(deg_inv, torch.matmul(adj_eye, deg_inv))

    return adj_hat
