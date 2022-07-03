import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import Tensor
from typing import NamedTuple


logger = logging.getLogger(__name__)


class Cora(NamedTuple):
    """Container for cora dataset."""
    df_nodes: pd.DataFrame
    df_edges: pd.DataFrame
    df_labels: pd.DataFrame
    num_nodes: int
    num_classes: int
    edges: list[tuple[int, int]]
    adj: Tensor
    nfeats: Tensor
    labels: Tensor
    labels_map: dict


def load_cora(path: Path) -> Cora:
    df_nodes = pd.read_csv(path / 'cora-nodes.csv')
    df_edges = pd.read_csv(path / 'cora-edges.csv')
    df_labels = pd.read_csv(path / 'cora-labels.csv')

    # reindex node ids to start at zero
    df_nodes = df_nodes.sort_values(by='paper_id')
    num_nodes = df_nodes.shape[0]
    df_nodes['idx'] = [i for i in range(num_nodes)]

    nfeats = df_nodes.iloc[:, 1:].to_numpy()
    nfeats = torch.from_numpy(nfeats).type(torch.float)
    logger.debug('node features:\n%s\n%s', nfeats, nfeats.shape)

    edges = []
    adj = torch.zeros(num_nodes, num_nodes)
    for row in df_edges.itertuples():
        src_id = df_nodes[df_nodes['paper_id'] == row.citing_paper_id]['idx'].item()
        dst_id = df_nodes[df_nodes['paper_id'] == row.cited_paper_id]['idx'].item()
        edges.append((src_id, dst_id))
        edges.append((dst_id, src_id))
        adj[src_id][dst_id] = 1
        adj[dst_id][src_id] = 1
    logger.debug('adjacency:\n%s\n%s', adj, adj.shape)

    label_map = np.sort(df_labels['class_label'].unique())
    label_map = {name: i for i, name in enumerate(label_map)}

    num_classes = len(label_map)
    labels = torch.zeros(num_nodes).type(torch.long)
    for row in df_labels.itertuples():
        node_id = df_nodes[df_nodes['paper_id'] == row.paper_id]['idx'].item()
        label_id = label_map[row.class_label]
        labels[node_id] = label_id
    logger.debug('labels:\n%s\n%s', labels, labels.shape)

    return Cora(
        df_nodes=df_nodes,
        df_edges=df_edges,
        df_labels=df_labels,
        num_nodes=num_nodes,
        num_classes=num_classes,
        edges=edges,
        adj=adj,
        nfeats=nfeats,
        labels=labels,
        labels_map=label_map)
