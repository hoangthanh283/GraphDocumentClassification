from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import load_data, accuracy
from data_loader import DatasetLoader, CLASSES
from models.gat import GAT, SpGAT, EGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

# Data params
parser.add_argument('--root_dir', type=str, default="./data/toyota_data", help='root path of data')
parser.add_argument('--train_file', type=str, default="train.txt", help='a text file that contains training json label paths')
parser.add_argument('--val_file', type=str, default="val.txt", help='a text file that contains validating json label paths')
parser.add_argument('--corpus_file', type=str, default="charset.txt", help='a file that contains all corpus')
parser.add_argument('--label_file', type=str, default="classification.xlsx", help='a excel file that contains true labels')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load train & val data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
train_kwargs = {
        'root_dir': args.root_dir,
        'data_file': args.train_file,
        'corpus_file': args.corpus_file,
        'label_file': args.label_file
    }
train_data = DatasetLoader(kwargs=train_kwargs, transform=True)
train_loader = DataLoader(train_data, shuffle=True)


val_kwargs = {
        'root_dir': args.root_dir,
        'data_file': args.val_file,
        'corpus_file': args.corpus_file,
        'label_file': args.label_file
    }
val_data = DatasetLoader(kwargs=val_kwargs, transform=True)
val_loader = DataLoader(val_data, shuffle=True)


# Model and optimizer
model = EGAT(node_feat=len(train_data.corpus),
            edge_feat=8,
            nclass=len(CLASSES),
            nhidden=args.hidden,
            dropout=args.dropout,
            alpha=args.alpha,
            nheads = args.nb_heads)
model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    t = time.time()
    model.train()
    loss_mean = []
    acc_mean = []
    for in_data in train_loader:
        optimizer.zero_grad()
        output = model(in_data)
        label = in_data['graph_lbl']

        loss_train = F.nll_loss(output, label)

        acc_train = accuracy(output, label)
        loss_mean.append(loss_train.data.item())
        acc_mean.append(acc_train)
        loss_train.backward()
        optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(np.mean(loss_mean)),
        'acc_train: {:.4f}'.format(np.mean(acc_mean)),
        'time: {:.4f}s'.format(time.time() - t))
    return np.mean(loss_mean)

def val(epoch):
    t = time.time()
    model.eval()
    loss_mean = []
    acc_mean = []
    for in_data in val_loader:
        output = model(in_data)
        label = in_data['graph_lbl']
        loss_val = F.nll_loss(output, label)

        acc_val = accuracy(output, label)
        loss_mean.append(loss_val.data.item())
        acc_mean.append(acc_val)

    print('Epoch: {:04d}'.format(epoch+1),
        'loss_val: {:.4f}'.format(np.mean(loss_mean)),
        'acc_val: {:.4f}'.format(np.mean(acc_mean)),
        'time: {:.4f}s'.format(time.time() - t))
    return np.mean(loss_mean)


if __name__ == "__main__":
    # Train model
    t_total = time.time()
    loss_values = []
    for epoch in range(args.epochs):
        train_loss = train(epoch)
        # val_loss = val(epoch)
        # loss_values.append(val_loss)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
