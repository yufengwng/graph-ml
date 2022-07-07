import sys
from pathlib import Path
sys.path.append(str(Path(__file__ + '/../../..').resolve()))
import gml  # Re-export library.


import matplotlib.pyplot as plt
import torch

from torch import Tensor
from typing import TypeAlias
_Loss: TypeAlias = torch.nn.Module
_Optimizer: TypeAlias = torch.optim.Optimizer


def gen_rand_edges(num_nodes: int, directed=False) -> list[tuple[int, int]]:
    edges = []
    for src in range(num_nodes):
        degree = torch.randint(1, 3, (1,)).item()
        neighbors = torch.randint(0, num_nodes, (degree,))
        for dst in neighbors:
            dst = dst.item()
            if src == dst:
                continue  # skip self-edges
            edges.append((src, dst))
            if not directed:
                edges.append((dst, src))
    return edges


def plot_graph(adj: Tensor, points: Tensor, colors: Tensor = None):
    plt.figure()
    for src, row in enumerate(adj):
        for dst, connected in enumerate(row):
            if connected == 0:
                continue
            point_src = points[src]
            point_dst = points[dst]
            xs = [point_src[0], point_dst[0]]
            ys = [point_src[1], point_dst[1]]
            plt.plot(xs, ys, linewidth=1, c='gray')
    num_nodes = adj.shape[0]
    area = (torch.zeros(num_nodes) + 10) ** 2
    xs = points[:, 0]
    ys = points[:, 1]
    plt.scatter(xs, ys, s=area, c=colors)
    plt.show()


def train(model: gml.GCN, g: gml.Graph, adj: Tensor, loss_fn: _Loss, optimizer: _Optimizer) -> Tensor:
    model.train()
    optimizer.zero_grad()
    pred = model(adj, g.ndata.feats)
    loss = loss_fn(pred, g.ndata.label)
    loss.backward()
    optimizer.step()
    return loss


def evaluate(model: gml.GCN, g: gml.Graph, adj: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        y_pred = model(adj, g.ndata.feats).argmax(dim=1)
        correct = (y_pred == g.ndata.label).type(torch.float).sum().item()
    accuracy = (correct / g.n_nodes) * 100
    return accuracy
