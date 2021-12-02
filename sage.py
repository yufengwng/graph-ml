import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from typing import List, Set, Tuple


class Graph(object):
    """A graph that encapsulates nodes, edges, and their features."""

    def __init__(self, edges: List[Tuple], features: Tensor):
        num_nodes = features.size(dim=0)
        adj_list = [set() for _ in range(num_nodes)]
        for v, u in edges:
            adj_list[v].add(u)
            adj_list[u].add(v)

        self.adj_list = adj_list
        self.features = features
        self.num_nodes = num_nodes
        self.dim_feats = features.size(dim=1)

    def nodes(self) -> List[int]:
        return list(range(self.num_nodes))


class GraphSage(nn.Module):
    """Based on Hamilton et al's 2017 GraphSAGE paper."""

    def __init__(self, dim_features: int, dim_embeds: int, sample_sizes: List[int] = None):
        super().__init__()

        if sample_sizes is None:
            sample_sizes = [25, 10]  # default k = 2

        layers = []
        dim_in = dim_features
        for size in sample_sizes:
            layers.append(SageLayer(size, dim_in, dim_embeds))
            dim_in = dim_embeds

        self.layers = nn.Sequential(*layers)
        self.dim_feats = dim_features
        self.dim_embeds = dim_embeds

    def forward(self, graph: Graph) -> Tensor:
        assert self.dim_feats == graph.dim_feats

        embeds_init = graph.features
        input = (graph, embeds_init)
        output = self.layers(input)
        embeds = output[1]

        return embeds


class SageLayer(nn.Module):
    """Computes node embeddings at depth k."""

    def __init__(self, sample_size: int, dim_embeds_in: int, dim_embeds_out: int):
        super().__init__()
        self.weights = nn.Linear(dim_embeds_in * 2, dim_embeds_out, bias=False)
        self.sampler = UniformSampler(sample_size)
        self.aggregator = MaxPoolAggregator(dim_embeds_in)

    def forward(self, input: Tuple[Graph, Tensor]) -> Tuple:
        graph, embeds_in = input

        neigh_embeds = []
        for node_idx in graph.nodes():
            # sample neighborhood of node
            samp_embeds = self.sampler.draw(graph, embeds_in, node_idx)
            # aggregate into single vector
            agg_embed = self.aggregator(samp_embeds)
            # collect embedding in same order
            neigh_embeds.append(agg_embed)
        neigh_embeds = torch.cat(neigh_embeds, dim=0)

        concat_embeds = torch.cat([embeds_in, neigh_embeds], dim=1)
        embeds_out = self.weights(concat_embeds)

        l2norm = torch.linalg.norm(embeds_out, dim=1, ord=2)
        l2norm = l2norm.reshape(len(embeds_out), 1)
        embeds_out = embeds_out.div(l2norm)

        # forward to next layer
        return (graph, embeds_out)


class UniformSampler(object):
    """Random uniform selection (with replacement) of neighbors."""

    def __init__(self, sample_size: int):
        self.sample_size = sample_size

    def draw(self, graph: Graph, embeds: Tensor, node_idx: int) -> Tensor:
        nb_set = graph.adj_list[node_idx]
        nb_indices = torch.LongTensor(list(nb_set))

        choices = torch.randint(len(nb_indices), (self.sample_size,))
        selection = nb_indices[choices]

        sampled = embeds[selection]
        if len(sampled) == 1:
            dim_embeds = embeds.size(dim=1)
            sampled = sampled.reshape(1, dim_embeds)

        return sampled


class MaxPoolAggregator(nn.Module):
    """Aggregate neighborhoods using element-wise max after MLP transformation."""

    def __init__(self, dim_embeds: int, bias=True):
        super().__init__()
        self.dim_embeds = dim_embeds
        self.weights = nn.Linear(dim_embeds, dim_embeds, bias=bias)

    def forward(self, embeds_neighbors: Tensor) -> Tensor:
        embeds_out = self.weights(embeds_neighbors)
        embeds_out = F.relu(embeds_out)
        max_out = embeds_out.max(dim=0).values
        return max_out.reshape(1, self.dim_embeds)


def train(model: GraphSage, graph: Graph, y: Tensor, loss_fn, optimizer):
    embeds = model(graph)
    loss = loss_fn(embeds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss: {loss:>7f}")


def evaluate(model: GraphSage, graph: Graph, y: Tensor):
    with torch.no_grad():
        embeds = model(graph)
        y_pred = embeds.argmax(dim=1)
        correct = (y_pred == y).type(torch.float).sum().item()

    accuracy = (correct / graph.num_nodes) * 100
    print(f"accuracy: {accuracy:>0.1f}%")
