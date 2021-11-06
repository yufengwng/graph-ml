import torch
from torch import nn


class GraphConvNet(nn.Module):
    """A graph convolutional network.

    Based on Kipf & Welling's 2017 GCN paper.
    """

    def __init__(self, adjacency_mat, num_layers: int,
                in_features: int, out_features: int,
                hidden_features=16, is_sparse=False):
        super().__init__()

        # stack:
        # - gcn layer [D x H]
        # - relu
        # - gcn layer [H x H]
        # - ...
        # - relu
        # - gcn layer [H x F]
        # - (softmax)

        shape = adjacency_mat.shape
        assert shape[0] == shape[1]
        assert num_layers > 0

        self.num_nodes = shape[0]
        self.num_layers = num_layers

        adj_eye = adjacency_mat + torch.eye(self.num_nodes)
        deg_mat = torch.diag(adj_eye.sum(dim=1))
        deg_inv = deg_mat.inverse().sqrt()
        self.adj_norm = deg_inv @ (adj_eye @ deg_inv)

        if is_sparse:
            self.adj_norm = self.adj_norm.to_sparse()

        layers = []
        if num_layers == 1:
            layers.append(GCNLayer(self.adj_norm, in_features, out_features))
        else:
            in_layer = GCNLayer(self.adj_norm, in_features, hidden_features)
            layers.append(in_layer)

            for _ in range(num_layers - 2):
                hidden = GCNLayer(self.adj_norm, hidden_features, hidden_features)
                layers.append(nn.ReLU())
                layers.append(hidden)

            out_layer = GCNLayer(self.adj_norm, hidden_features, out_features)
            layers.append(nn.ReLU())
            layers.append(out_layer)

        self.prop_rule = nn.Sequential(*layers)

    def forward(self, x):
        return self.prop_rule(x)


class GCNLayer(nn.Module):
    def __init__(self, adjacency_norm, in_features: int, out_features: int):
        super().__init__()
        self.adj_norm = adjacency_norm
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.adj_norm @ self.linear(x)


def train(X, y, model, loss_fn, optimizer):
    # pytorch will do the softmax for us
    # so just pass the raw output to loss
    pred = model(X)

    # labels can be array of indices
    # don't need to be one-hot
    loss = loss_fn(pred, y)

    # reset gradients to avoid double-counting (?)
    optimizer.zero_grad()

    # backpropagate (how does it know where the params are??)
    # might have to do with the autograd objects
    loss.backward()

    # adjust the parameters
    optimizer.step()

    print(f"loss: {loss:>7f}")


def evaluate(X, y, model):
    with torch.no_grad():
        pred = model(X)
        y_pred = pred.argmax(dim=1)
        correct = (y_pred == y).type(torch.float).sum().item()
    num_nodes = X.shape[0]
    accuracy = (correct / num_nodes) * 100
    print(f"accuracy: {accuracy:>0.1f}%")