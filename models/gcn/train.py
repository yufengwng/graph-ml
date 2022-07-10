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
        hidden_dim=args.n_hidden,
        dropout=args.dropout)
model = model.to(device)
logger.debug('model:\n%s', model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

t_start = time.perf_counter()
for epoch in range(args.n_epochs):
    logger.info('epoch %-3d', epoch)
    loss = support.train(model, g, adj, criterion, optimizer)
    logger.info('loss: %.7f', loss)
    accuracy = support.evaluate(model, g, adj)
    logger.info('accuracy: %0.1f%%', accuracy)
t_elapsed = time.perf_counter() - t_start
logger.info('finished training, total elapsed: %.4fs', t_elapsed)

if args.save:
    torch.save(model.state_dict(), args.save)
    logger.info('saved model to file: %s', args.save)
