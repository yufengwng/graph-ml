import torch
import torch.nn.functional as fn

from torch import nn
from torch import Tensor
from typing import List, Tuple


RELU_NEG_SLOPE = 0.2


class Graph(object):
    """A simple container for a graph."""

    def __init__(self, edges: List[Tuple], features: Tensor):
        self.num_nodes = features.shape[0]
        self.dim_feats = features.shape[1]
        self.features = features

        adj_list = [set() for _ in range(self.num_nodes)]
        for v, u in edges:
            adj_list[v].add(u)
            adj_list[u].add(v)
        self.adj_list = adj_list

    def nodes(self) -> List[int]:
        return list(range(self.num_nodes))

    def neighborhood(self, node) -> List[int]:
        neighbors = set(self.adj_list[node])
        neighbors.add(node)  # self connection
        return list(neighbors)


class GAT(nn.Module):
    """A graph attention network (GAT) based on Velickovic et al 2018."""

    def __init__(self, head_sizes: List[int], head_feats: int, dim_features: int, dim_embeds: int):
        super().__init__()

        num_layers = len(head_sizes)
        assert num_layers >= 1

        layers = []
        if num_layers == 1:
            layer = AttentionLayer(self, head_sizes[0], dim_embeds, dim_features, concat=False)
            layers.append(layer)
        else:
            layer = AttentionLayer(self, head_sizes[0], head_feats, dim_features, concat=True)
            dim_in = layer.dim_embeds_out
            layers.append(layer)

            for i in range(1, num_layers - 2):
                layer = AttentionLayer(self, head_sizes[i], head_feats, dim_in, concat=True)
                dim_in = layer.dim_embeds_out
                layers.append(nn.ELU())
                layers.append(layer)

            layer = AttentionLayer(self, head_sizes[-1], dim_embeds, dim_in, concat=False)
            layers.append(nn.ELU())
            layers.append(layer)

        self.seq = nn.Sequential(*layers)
        self.dim_feats = dim_features

        # dynamically set this attribute as a way of passing the graph to submodules
        self.graph = None

    def forward(self, graph: Graph) -> Tensor:
        assert self.dim_feats == graph.dim_feats

        self.graph = graph
        embeds = graph.features
        embeds = self.seq(embeds)
        self.graph = None

        return embeds


class AttentionLayer(nn.Module):
    """A single graph attentional layer, mostly for handling multi-head aggregation."""

    def __init__(self, gat: GAT, num_heads: int, head_feats: int, dim_embeds_in: int, concat=True):
        super().__init__()
        assert num_heads >= 1

        heads = []
        for _ in range(num_heads):
            head = AttentionHead(gat, dim_embeds_in, head_feats)
            heads.append(head)
        self.heads = nn.ModuleList(heads)

        self.concat = concat
        self.dim_embeds_in = dim_embeds_in
        self.dim_embeds_out = num_heads * head_feats if self.concat else head_feats

    def forward(self, embeds_in: Tensor) -> Tensor:
        head_outs = [head(embeds_in) for head in self.heads]
        if self.concat:
            embeds_out = torch.cat(head_outs, dim=1)
            # shape: [N x HF']
        else:
            embeds_out = torch.stack(head_outs, dim=0)
            embeds_out = embeds_out.mean(dim=0)
            # shape: [N x C] where C == F'
        return embeds_out


class AttentionHead(nn.Module):
    """Computes representation by applying attention mechanism on neighborhood."""

    def __init__(self, gat: GAT, dim_embeds_in: int, dim_embeds_out: int):
        super().__init__()

        self.proj_weights = nn.Linear(dim_embeds_in, dim_embeds_out, bias=False)
        self.attn_weights = nn.Linear(2 * dim_embeds_out, 1, bias=False)

        # hack to avoid registering as submodule which causes recursion issues
        object.__setattr__(self, "_gat", gat)

    def forward(self, embeds_in: Tensor) -> Tensor:
        proj = self.proj_weights(embeds_in)
        # shape: [N x F']

        node_embeds = []
        num_features = proj.shape[1]
        graph = self._gat.graph

        for i in graph.nodes():
            proj_neigh = proj[graph.neighborhood(i)]
            # shape: [N_i x F']

            num_neighbors = proj_neigh.shape[0]
            proj_self = proj[i].repeat(num_neighbors, 1)
            # shape: [N_i x F']

            proj_concat = torch.cat([proj_self, proj_neigh], dim=1)
            # shape: [N_i x 2F']

            coeffs = self.attn_weights(proj_concat)
            coeffs = fn.leaky_relu(coeffs, negative_slope=RELU_NEG_SLOPE)
            coeffs = fn.softmax(coeffs, dim=0)
            # shape: [N_i x 1]

            embeds = proj_neigh * coeffs  # element-wise
            embeds = embeds.sum(dim=0)
            embeds = embeds.reshape(1, num_features)
            # shape: [1 x F']

            node_embeds.append(embeds)

        embeds_out = torch.cat(node_embeds, dim=0)
        # shape: [N x F']

        return embeds_out


def train(model: GAT, graph: Graph, y: Tensor, loss_fn, optimizer):
    embeds = model(graph)
    # xentropy automatically performs softmax
    loss = loss_fn(embeds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss: {loss:>7f}")


def evaluate(model: GAT, graph: Graph, y: Tensor):
    with torch.no_grad():
        embeds = model(graph)
        logits = fn.softmax(embeds, dim=1)
        y_pred = logits.argmax(dim=1)
        correct = (y_pred == y).type(torch.float).sum().item()
    accuracy = (correct / graph.num_nodes) * 100
    print(f"accuracy: {accuracy:>0.1f}%")
