"""Module for operating on graphs."""

import torch

from .typing import Device, EdgeList, Tensor
from .utils import Namespace


class Graph(object):
    """A simple representation for static graphs.

    When constructing a graph, the list of edges should use node indices
    starting from zero and is directed as `(source, destination)`.

    # Examples

    ```
    >>> import torch
    >>> from gml.graph import Graph
    >>>
    >>> g = Graph(3, [(0, 1), (1, 2)])
    >>> g
    Graph(n_nodes=3, n_edges=2, device=cpu,
          ndata=Namespace(_dim=3),
          edata=Namespace(_dim=2))
    ```
    ----------------------------------------------------------------------
    """

    def __init__(self, num_nodes: int, edges: EdgeList, device: Device = 'cpu'):
        """
        # Parameters

        * `num_nodes` - Number of nodes in graph
        * `edges` - List of (src, dst) edges
        * `device` - Where to allocate tensors
        """
        adj_list = [[] for _ in range(num_nodes)]
        for src, dst in edges:
            adj_list[src].append(dst)

        offsets = [0]
        indices = []
        for src in range(num_nodes):
            neighbors = adj_list[src]
            neighbors = sorted(neighbors)
            indices.extend(neighbors)
            offsets.append(len(indices))

        assert (len(offsets) == num_nodes + 1)
        assert (offsets[-1] == len(indices))

        self.device = device
        self.n_nodes = num_nodes
        self.n_edges = len(indices)
        self.ndata = Namespace(self.n_nodes)
        self.edata = Namespace(self.n_edges)
        self._csr_offsets = torch.tensor(offsets, dtype=torch.int, device=device)
        self._csr_indices = torch.tensor(indices, dtype=torch.int, device=device)

    def __repr__(self) -> str:
        return f"Graph(n_nodes={self.n_nodes}, n_edges={self.n_edges}, device={self.device},\n" \
               f"      ndata={self.ndata},\n" \
               f"      edata={self.edata})" \

    def adj(self) -> Tensor:
        """Returns the dense adjacency matrix of this graph."""
        n_nodes = self.n_nodes
        adj_mtx = torch.zeros(n_nodes, n_nodes, dtype=torch.int, device=self.device)
        for src in range(n_nodes):
            idx = self._csr_offsets[src:src+2]
            neighbors = self._csr_indices[idx[0]:idx[1]]
            for dst in neighbors:
                adj_mtx[src, dst] = 1
        return adj_mtx


class NeighborSampler(object):

    def __init__(self):
        pass
