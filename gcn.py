import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

import time
import random
import numpy as np
from collections import defaultdict

class gs_block(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, 
        policy='mean', gcn=False, num_sample=10
    ):
        super().__init__()
        self.gcn = gcn
        self.policy=policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.num_sample = num_sample
        self.weight = nn.Parameter(torch.FloatTensor(
            embed_dim, 
            self.feat_dim if self.gcn else 2*self.feat_dim
        ))
        init.xavier_uniform_(self.weight)

    def forward(self, x, Adj):
        neigh_feats = self.aggregate(x, Adj)
        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined,2,1)
        return combined
    def aggregate(self,x, Adj):
        adj=Variable(Adj).to(Adj.device)
        if not self.gcn:
            n=len(adj)
            adj = adj-torch.eye(n).to(adj.device)
        if self.policy=='mean':
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            to_feats = mask.mm(x)
        elif self.policy=='max':
            indexs = [i.nonzero() for i in adj==1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1:
                    to_feats.append(feat.view(1, -1))
                else:
                    to_feats.append(torch.max(feat,0)[0].view(1, -1))
            to_feats = torch.cat(to_feats, 0)
        return to_feats