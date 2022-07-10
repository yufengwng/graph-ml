import argparse
import time
from pathlib import Path

import torch

import support
from support import gml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(Path(__file__).name)
    parser.add_argument('-d', '--data', type=str, required=True,
        help='Dataset to use')
    parser.add_argument('--path', type=str, required=True,
        help='Directory path to datasets')
    parser.add_argument('--model', type=str, required=True,
        help='File path to trained model')
    parser.add_argument('--n-layers', type=int, default=2,
        help='Number of layers (default: 2)')
    parser.add_argument('--n-hidden', type=int, default=16,
        help='Number of hidden features (default: 16)')
    parser.add_argument('--gpu', type=int, default=-1,
        help='Index of gpu to use (default: -1 for cpu)')
    parser.add_argument('--seed', type=int, default=-1,
        help='Random seed (default: -1 for no seed)')
    parser.add_argument('--debug', action='store_true',
        help='Enable debug logging')
    return parser.parse_args()


args = parse_args()
logger = support.setup_logging(args)
support.setup_seed(args, logger)

device = support.build_device(args, logger)
g, n_classes = support.load_data(device, args, logger)
n_feats = g.ndata.feats.shape[1]

adj = g.adj()
adj = gml.gcn.renormalize_adjacency(adj)
logger.info('graph:\n%s', g)

model = gml.GCN(args.n_layers,
        in_dim=n_feats,
        out_dim=n_classes,
        hidden_dim=args.n_hidden)
model = model.to(device)

logger.info('loading model parameters...')
keys = model.load_state_dict(torch.load(args.model))
logger.info('incompatible keys: %s', keys)

model = model.eval()
logger.debug('model:\n%s', model)

with torch.no_grad():
    t_start = time.perf_counter_ns()
    model(adj, g.ndata.feats)
    t_elapsed = time.perf_counter_ns() - t_start
logger.info('finished inference, total elapsed: %.4fms', t_elapsed / 1e6)
