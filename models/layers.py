import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class EdgeGraphAttentionLayer(nn.Module):
    def __init__(self, node_in_fts, ed_in_fts, node_out_fts, ed_out_fts, dropout=0.0, alpha=0.0, concat=True):
        super(EdgeGraphAttentionLayer, self).__init__()
        self.node_in_fts = node_in_fts
        self.node_out_fts = node_out_fts
        self.ed_in_fts = ed_in_fts
        self.ed_out_fts = ed_out_fts
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.triplet_transform = Parameter(torch.zeros(
            size=(2*self.node_in_fts + self.ed_in_fts, self.node_out_fts)))

        self.edge_transform = Parameter(torch.zeros(
            size=(self.node_out_fts, self.ed_out_fts)))

        self.att = Parameter(torch.Tensor(self.node_out_fts, 1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.triplet_transform.data, gain=1.414)
        nn.init.xavier_uniform_(self.edge_transform.data, gain=1.414)
        nn.init.xavier_uniform_(self.att.data, gain=1.414)

    def forward(self, node_fts_txt, edge_fts):
        N = node_fts_txt.size(1)
        triplet = torch.cat([node_fts_txt.repeat(1, 1, N).view(1, N*N, -1), \
            edge_fts, node_fts_txt.repeat(1, N, 1)], dim=-1)
        
        # N^2-fts x fts-out_fts = N^2-out_fts
        new_hij = torch.matmul(triplet, self.triplet_transform)
        
        # N^2-fts x edge_fts = N^2-edge_fts
        new_edge_fts = torch.matmul(new_hij, self.edge_transform)
        new_hij = new_hij.view(1, N, -1, self.node_out_fts)  # 1-N-N-fts

        attention = torch.matmul(new_hij, self.att).squeeze(-1)  # N-N-fts x fxs-1 = N-N
        attention = F.leaky_relu(attention, self.alpha)  # N-N
        attention = F.softmax(attention, dim=-1)  # 1-N-N
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention.view(1, N, 1, N), new_hij)
        new_x = h_prime.squeeze(2)

        return new_x, new_edge_fts

class GraphAttentionLayer(nn.Module):
    """ Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
  
class SpGraphAttentionLayer(nn.Module):
    """ Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime