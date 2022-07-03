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
parser.add_argument('--n-epochs', type=int, default=200,
    help='Number of training epochs (default: 200)')
parser.add_argument('--n-layers', type=int, default=2,
    help='Number of layers (default: 2)')
parser.add_argument('--n-hidden', type=int, default=16,
    help='Number of hidden features (default: 16)')
parser.add_argument('--lr', type=float, default=0.01,
    help='Model learning rate (default: 0.01)')
parser.add_argument('--dropout', type=float, default=0.5,
    help='Dropout probability (default: 0.5)')
parser.add_argument('--gpu', type=int, default=-1,
    help='Index of gpu to use (default: -1 for cpu)')
parser.add_argument('--seed', type=int, default=-1,
    help='Random seed (default: -1 for no seed)')
parser.add_argument('--save', type=str, default='',
    help='File path to save trained model')
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
        hidden_dim=args.n_hidden,
        dropout=args.dropout)
model = model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
logger.debug('model:\n%s', model)


t_start = time.perf_counter()
for epoch in range(args.n_epochs):
    logger.info('epoch %-3d', epoch)
    loss = support.train(model, g, adj, loss_fn, optimizer)
    logger.info('loss: %.7f', loss)
    accuracy = support.evaluate(model, g, adj)
    logger.info('accuracy: %0.1f%%', accuracy)
logger.info('finished training, total elapsed: %.4fs', time.perf_counter() - t_start)


if args.save:
    torch.save(model.state_dict(), args.save)
    logger.info('saved model to file: %s', args.save)
