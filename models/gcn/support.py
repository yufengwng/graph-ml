import logging
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import loader


# Re-export library.
sys.path.append(str(Path(__file__ + '/../../..').resolve()))
import gml


from argparse import Namespace
from logging import Logger
from torch import Tensor
from typing import Optional, TypeAlias
from gml.typing import Device
LossFn: TypeAlias = torch.nn.Module
Optimizer: TypeAlias = torch.optim.Optimizer


def setup_logging(args: Namespace) -> Logger:
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s.%(msecs)03d|%(name)s|%(levelname)s|%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.info('args: %s', args)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    return logger


def setup_seed(args: Namespace, logger: Optional[Logger] = None):
    if args.seed > -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if logger is not None:
            logger.info('configured random seed with: %d', args.seed)


def build_device(args: Namespace, logger: Optional[Logger] = None) -> Device:
    if torch.cuda.is_available() and args.gpu > -1:
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device('cpu')
    if logger is not None:
        logger.info('using device: %s', device)
    return device


def load_data(device: Device, args: Namespace, logger: Optional[Logger] = None) -> tuple[gml.Graph, int]:
    if logger is not None:
        logger.info('loading dataset...')
    if args.data == 'cora':
        data = loader.load_cora(Path(args.path))
    else:
        print('currently only supports cora dataset')
        sys.exit(0)
    g = gml.Graph(data.num_nodes, data.edges, device=device)
    g.ndata.feats = data.nfeats
    g.ndata.label = data.labels
    return g, data.num_classes


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


def train(model: gml.GCN, g: gml.Graph, adj: Tensor, criterion: LossFn, optimizer: Optimizer) -> Tensor:
    model.train()
    pred = model(adj, g.ndata.feats)
    loss = criterion(pred, g.ndata.label)
    optimizer.zero_grad()
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
