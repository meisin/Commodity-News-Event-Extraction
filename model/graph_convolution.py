"""
Graph Convolution Network (GCN) over dependency parse tree
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class GCN(nn.Module):
    """ A GCN module operated on dependency graphs. """
    """ Modified from gcn.py from 'Graph Convolution over Pruned Dependency Trees for Relation Extraction' """
    
    def __init__(self, hidden_dim, num_layers, total_dim):   
        super(GCN, self).__init__()
        self.layers = num_layers
        self.hidden_dim = hidden_dim    
        self.in_dim = total_dim
        self.useRNN = True   ## Ablation study
        
        # for Ablation study - rnn layer
        if self.useRNN:
            input_size = self.in_dim
            self.rnn_hidden = 200
            self.rnn_layers = 1
            self.rnn = nn.LSTM(input_size, self.rnn_hidden, self.rnn_layers, batch_first=True, bidirectional=True)
            self.in_dim = self.rnn_hidden * 2
        
        # GCN layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.hidden_dim   
            self.W.append(nn.Linear(input_dim, self.hidden_dim))         
            
    def encode_with_rnn(self, rnn_inputs, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.rnn_hidden, self.rnn_layers)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        return rnn_outputs        
    
    def forward(self, adj, gcn_inputs):
        
        # for Ablation study - rnn layer
        if self.useRNN:
            gcn_inputs = self.encode_with_rnn(gcn_inputs, gcn_inputs.size()[0])       
            
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1   ## [batch_size, maxlen, 1]  - suming up at dimension = 2
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2) ## [batch_size, maxlen, 1] - suming up at dimension 1 and 2
                                    ## mask is to indicate if the specific word is needed, not needed where word is not in the adj link

        #### for Ablation study - turning off graph dependency matrix
        #adj = torch.zeros_like(adj)       
        
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # add self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            
            gcn_inputs = gAxW

        return gcn_inputs, mask

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    """ Initialize h0 and c0 """
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0