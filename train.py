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

from utils import accuracy
from data_loader import DatasetLoader, CLASSES
from models.gat import GAT, SpGAT, EGAT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(args):
    # Load train & val data
    # adj, features, labels, idx_train, idx_val, idx_test = load_data()
    train_kwargs = {
            'root_dir': args.root_dir,
            'data_file': args.train_file,
            'corpus_file': args.corpus_file,
            'label_file': args.label_file
        }
    train_data = DatasetLoader(kwargs=train_kwargs, transform=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_kwargs = {
            'root_dir': args.root_dir,
            'data_file': args.val_file,
            'corpus_file': args.corpus_file,
            'label_file': args.label_file
        }
    val_data = DatasetLoader(kwargs=val_kwargs, transform=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)


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
    
    best_loss = 1000
    best_acc = 0.0

    for epoch in range(0, args.epochs):
        model.train()
        train_loss_mean = []
        train_acc_mean = []
        start_time = time.time()
        for in_data in train_loader:
            optimizer.zero_grad()
            output = model(in_data)
            label = in_data['graph_lbl'].to(DEVICE)
            loss_train = F.nll_loss(output, label)
            loss_train.backward()
            optimizer.step()

            acc_train = accuracy(output, label)
            train_loss_mean.append(loss_train.data.item())
            train_acc_mean.append(acc_train)

        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(np.mean(train_loss_mean)),
            'acc_train: {:.4f}'.format(np.mean(train_acc_mean)),
            'time: {:.4f}s'.format(time.time() - start_time))

        if epoch == args.patience:
            model.eval()
            val_loss_mean = []
            val_acc_mean = []
            for in_data in val_loader:
                output = model(in_data)
                label = in_data['graph_lbl']
                loss_val = F.nll_loss(output, label)

                acc_val = accuracy(output, label)
                val_loss_mean.append(loss_val.data.item())
                val_acc_mean.append(acc_val)

            print("*"*20)
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_val: {:.4f}'.format(np.mean(val_loss_mean)),
                'acc_val: {:.4f}'.format(np.mean(val_acc_mean)))

            if (np.mean(val_acc_mean) > best_acc and np.mean(val_loss_mean)):
                torch.save({
                    "state_dict": model.state_dict(),
                    "configs": args,
                    "epoch": epoch,
                    "train_acc": np.mean(train_loss_mean),
                    "val_acc": np.mean(val_loss_mean),
                }, "{0}_epoch_{1}.pt".format(args.save_path, epoch))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    # Data params 
    parser.add_argument('--root_dir', type=str, default="./data/toyota_data", help='root path of data')
    parser.add_argument('--save_path', type=str, default="./weights", help='path to save model')
    parser.add_argument('--train_file', type=str, default="train.txt", help='a text file that contains training json label paths')
    parser.add_argument('--val_file', type=str, default="val.txt", help='a text file that contains validating json label paths')
    parser.add_argument('--corpus_file', type=str, default="charset.txt", help='a file that contains all corpus')
    parser.add_argument('--label_file', type=str, default="classification.xlsx", help='a excel file that contains true labels')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    main(args)
    