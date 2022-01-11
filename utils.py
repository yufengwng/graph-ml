import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch import Tensor
from pandas import DataFrame
from typing import List, NamedTuple, Tuple


def gen_edges(num_nodes: int) -> List:
    edges = []
    for v in range(num_nodes):
        degree = torch.randint(1, 3, (1,)).item()
        neighbors = torch.randint(0, num_nodes, (degree,))
        for u in neighbors:
            if u == v:
                continue  # skip self-edges
            edges.append((v, u.item()))
    return edges


def make_adj_mat(num_nodes: int, undirected_edges: List[Tuple]) -> Tensor:
    adj = torch.zeros(num_nodes, num_nodes)
    for v, u in undirected_edges:
        adj[v][u] = 1
        adj[u][v] = 1
    return adj


def plot_graph(adj: Tensor, points: Tensor, colors: Tensor = None):
    for v, row in enumerate(adj):
        for u, connected in enumerate(row):
            if connected == 0:
                continue
            point_v = points[v]
            point_u = points[u]
            xs = [point_v[0], point_u[0]]
            ys = [point_v[1], point_u[1]]
            plt.plot(xs, ys, linewidth=1, c='gray')
    num_nodes = adj.shape[0]
    area = (torch.zeros(num_nodes) + 10) ** 2
    xs = points[:, 0]
    ys = points[:, 1]
    plt.scatter(xs, ys, s=area, c=colors)
    plt.show()


def plot_pca(features: Tensor, colors: Tensor = None):
    svd = torch.pca_lowrank(features)
    eigenvecs = svd[2]

    components = features @ eigenvecs[:, :2]
    xs = components[:, 0]
    ys = components[:, 1]

    area = (torch.zeros(features.shape[0]) + 10) ** 2
    plt.scatter(xs, ys, s=area, c=colors)
    plt.show()


class Cora(NamedTuple):
    df_nodes: DataFrame
    df_edges: DataFrame
    df_labels: DataFrame
    num_nodes: int
    num_classes: int
    adj: Tensor
    nodes: Tensor
    edges: List[Tuple]
    labels: Tensor
    labels_map: dict


def load_cora(debug=False) -> Cora:
    df_nodes = pd.read_csv('data/cora-nodes.csv')
    df_edges = pd.read_csv('data/cora-edges.csv')
    df_labels = pd.read_csv('data/cora-labels.csv')

    # setup nodes
    df_nodes = df_nodes.sort_values('paper_id')
    nodes = df_nodes.iloc[:, 1:].to_numpy()
    nodes = torch.from_numpy(nodes).type(torch.float)
    num_nodes = nodes.size(dim=0)
    if debug:
        print(f"nodes: {nodes.shape}")
        print(nodes)
        print()

    # setup edges
    edges = []
    adj = torch.zeros(num_nodes, num_nodes)
    for row in df_edges.itertuples():
        src_id = row.citing_paper_id
        dst_id = row.cited_paper_id
        src_idx = df_nodes.index[df_nodes['paper_id'] == src_id].item()
        dst_idx = df_nodes.index[df_nodes['paper_id'] == dst_id].item()
        edges.append((src_idx, dst_idx))
        adj[src_idx][dst_idx] = 1
        adj[dst_idx][src_idx] = 1
    if debug:
        print(f"adjacency: {adj.shape}")
        print(adj)
        print()

    # setup labels
    label_map = sorted(df_labels['class_label'].unique())
    label_map = {name: i for i, name in enumerate(label_map)}

    num_classes = len(label_map)
    labels = torch.zeros(num_nodes).type(torch.long)
    for row in df_labels.itertuples():
        paper_id = row.paper_id
        node_id = df_nodes.index[df_nodes['paper_id'] == paper_id].item()
        label_id = label_map[row.class_label]
        labels[node_id] = label_id
    if debug:
        print(f"labels: {labels.shape}")
        print(labels[:15], '...')

    return Cora(
        df_nodes, df_edges, df_labels,
        num_nodes, num_classes, adj,
        nodes, edges, labels, label_map)
