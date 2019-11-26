import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.layers import (
    GraphAttentionLayer, 
    SpGraphAttentionLayer,
    EdgeGraphAttentionLayer
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class EGAT(nn.Module):
    def __init__(self, node_feat, edge_feat, nclass, nhidden, dropout, alpha, nheads):
        super(EGAT, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(edge_feat, 1)

        self.attentions = [GraphAttentionLayer(\
            node_feat, nhidden, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(\
            nhidden * nheads, nhidden, dropout=dropout, alpha=alpha, concat=False)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(nhidden, nhidden // 2),
            nn.Dropout(p=dropout),
            nn.Linear(nhidden // 2, nclass)
        )

        self.activation = nn.LeakyReLU(alpha)

    def forward(self, in_data):
        """ Node embedding - Edge embedding """
        node_fts_txt = in_data['node_fts_txt'].to(DEVICE)
        node_fts_loc = in_data['node_fts_loc'].to(DEVICE)

        node_fts = node_fts_txt
        edge_fts = in_data['edge_fts'].to(DEVICE)

        edges = self.linear(edge_fts).squeeze(-1)
        edges = F.elu(edges)

        nodes = F.dropout(node_fts, self.dropout, training=self.training)
        edges = F.dropout(edges, self.dropout, training=self.training)

        x = torch.cat([att(nodes, edges) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.elu(self.out_att(x, edges))
        x = torch.max(x, dim=1)[0]
        x = self.fc(x)
        x = self.activation(x)
        
        return F.log_softmax(x, dim=1)