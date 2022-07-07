import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import loader
import support
from support import gml


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
args = parser.parse_args()
del parser


logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d|%(name)s|%(levelname)s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.info('args: %s', args)
if args.debug:
    logger.setLevel(logging.DEBUG)


if args.seed > -1:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.info('configured random seed with: %d', args.seed)

if torch.cuda.is_available() and args.gpu > -1:
    device = torch.device('cuda', args.gpu)
else:
    device = torch.device('cpu')
logger.info('using device: %s', device)


logger.info('loading dataset...')
if args.data == 'cora':
    data = loader.load_cora(Path(args.path))
else:
    print('currently only supports cora dataset')
    sys.exit(0)

n_classes = data.num_classes
n_feats = data.nfeats.shape[1]
g = gml.Graph(data.num_nodes, data.edges, device=device)
g.ndata.feats = data.nfeats
g.ndata.label = data.labels
del data

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
del keys

model = model.eval()
logger.debug('model:\n%s', model)


with torch.no_grad():
    t_start = time.perf_counter_ns()
    model(adj, g.ndata.feats)
    t_elapsed = time.perf_counter_ns() - t_start
logger.info('finished inference, total elapsed: %.4fms', t_elapsed / 1e6)
