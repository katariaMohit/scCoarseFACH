import os.path as osp
import torch
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric import utils
from torch_geometric.nn import Sequential
from torch.nn import Linear
from torch_geometric.nn import SAGEConv

from sklearn.metrics import normalized_mutual_info_score as NMI

from graph_clustering_helper_just_balance import just_balance_pool
from torch_geometric.utils import to_dense_adj


class Net(torch.nn.Module):
    def __init__(self, 
                 mp_units,
                 mp_act,
                 in_channels, 
                 n_clusters, 
                 mlp_units=[],
                 mlp_act="Identity"):
        super().__init__()
        
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # Message passing layers
        mp = [
            (GCNConv(in_channels, mp_units[0], normalize=False, cached=False), 'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            mp.append((GCNConv(mp_units[i], mp_units[i+1], normalize=False, cached=False), 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]
        
        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))
        

    def forward(self, x, edge_index, edge_weight):
        
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight)
        
        # Cluster assignments (logits)
        s = self.mlp(x)
        
        # Compute loss
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        # print("adj shape ",adj.shape, " x shape ",x.shape)
        _, _, b_loss = just_balance_pool(x, adj, s)
        
        return torch.softmax(s, dim=-1), b_loss
    

class Sage(torch.nn.Module):
    def __init__(self, 
                 mp_units,
                 mp_act,
                 in_channels, 
                 n_clusters, 
                 mlp_units=[],
                 mlp_act="Identity"):
        super().__init__()
        
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # Message passing layers (using GraphSAGEConv instead of GCNConv)
        mp = [
            (SAGEConv(in_channels, mp_units[0]), 'x, edge_index -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            mp.append((SAGEConv(mp_units[i], mp_units[i+1]), 'x, edge_index -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index', mp)
        out_chan = mp_units[-1]
        
        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))

    def forward(self, x, edge_index):
        # Propagate node feats
        x = self.mp(x, edge_index)
        
        # Cluster assignments (logits)
        s = self.mlp(x)
        
        # Compute loss
        adj = to_dense_adj(edge_index)
        # print(x.shape)
        # print(adj.shape)
        # print(s.shape)
        _, _, b_loss = just_balance_pool(x, adj, s)
        
        return torch.softmax(s, dim=-1), b_loss