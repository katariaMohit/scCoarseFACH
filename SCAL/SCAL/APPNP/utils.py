from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import to_dense_adj
from SCAL.SCAL.APPNP.coarsening_utils import *
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull

import numpy as np
import random
import math

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def extract_components(H):

        if H.A.shape[0] != H.A.shape[1]:
            H.logger.error('Inconsistent shape to extract components. '
                           'Square matrix required.')
            return None

        # if H.is_directed():
        #     raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []
        visited = np.zeros(H.A.shape[0], dtype=bool)

        while not visited.all():
            stack = set([np.nonzero(~visited)[0][0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                      if not visited[idx]]))

            comp = sorted(comp)
            G = H.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        return graphs


def coarsening(data, coarsening_ratio, coarsening_method):
    # if dataset == 'dblp':
    #     dataset = CitationFull(root='./dataset', name=dataset)
    # elif dataset == 'Physics':
    #     dataset = Coauthor(root='./dataset/Physics', name=dataset)
    # else:
    #     dataset = Planetoid(root='./dataset', name=dataset)
    # data = dataset[0]


    # ### Give the joint probability matrix in place of adjacency matrix --------

    # W = to_dense_adj(data.edge_index)[0]
    # print(W)
    # print(W.shape)
    # #np.fill_diagonal(W,1)

    # features_phi = W

    # # Pick projectors
    # projector_matrix = []
    # number_of_projectors = math.ceil(math.sqrt(features_phi.shape[0]))*50
    # for i in range(number_of_projectors):
    #     projector_matrix.append(np.array(features_phi[random.randint(0,features_phi.shape[0]-1)]))

    # projector_matrix = np.array(projector_matrix)
    # print("projector matrix size ",np.shape(projector_matrix))

    # # Find joint Probabilities

    # k_matrix = np.zeros((np.shape(features_phi)[0],np.shape(features_phi)[0]))

    # temp_counter = 0
    # for projector in projector_matrix:
    #     if temp_counter%10 == 0:
    #         print("new counter ",temp_counter)
    #     temp_counter += 1

    #     p_values = np.dot(projector,features_phi)
    #     temp1 = np.amin(p_values)
    #     temp2 = np.amax(p_values)
    #     num_of_supernodes = (int)(k_matrix.shape[0]/2)
    #     bin_width =  (temp2 - temp1)/num_of_supernodes
    #     p_values = np.floor((p_values - temp1)/bin_width)
        
    #     for j in range(num_of_supernodes):
    #         jth_supernode_indices = [i for i, x in enumerate(p_values) if x == j]
    #         #print(" j is ",j," jth_supernode_indices ",jth_supernode_indices)
    #         for k in range(len(jth_supernode_indices) - 1):
    #             counter = 0
    #             while(k+counter < len(jth_supernode_indices)):
    #                 #print("two values two be updated is ",jth_supernode_indices[k+counter],jth_supernode_indices[k],k,counter)
    #                 k_matrix[jth_supernode_indices[k]][jth_supernode_indices[k+counter]] += 1
    #                 if counter != 0:
    #                     k_matrix[jth_supernode_indices[k+counter]][jth_supernode_indices[k]] += 1
    #                 counter += 1

    #         if len(jth_supernode_indices) != 0:
    #             k_matrix[jth_supernode_indices[-1]][jth_supernode_indices[-1]] += 1

    # my_graph_adjacency = torch.from_numpy(k_matrix/projector_matrix.shape[0])


    # # # Find joint Probabilities

    # # p_matrix = np.zeros((np.shape(features_phi)[0],number_of_projectors))

    # # for i in range(np.shape(features_phi)[0]):
    # #     projectors_values = np.dot(projector_matrix,features_phi[i,:])
        
    # #     temp1 = np.amin(projectors_values)
    # #     temp2 = np.amax(projectors_values)

    # #     bin_width =  (temp2 - temp1)/800
    # #     #print(temp1,temp2,bin_width)
        
    # #     p_matrix[i] = np.floor((projectors_values - temp1)/bin_width)
        

    # # k_matrix = np.zeros((np.shape(features_phi)[0],np.shape(features_phi)[0]))

    # # print(" p matrix calculation done ",p_matrix.shape)
    # # for i in range(number_of_projectors):
    # #     if i%10 == 0:
    # #         print(i)
    # #     for j in range(np.shape(features_phi)[0]):
    # #         for k in range(np.shape(features_phi)[0]):
    # #             if p_matrix[j][i] == p_matrix[k][i]:
    # #                 k_matrix[j][k] += 1

    # # k_matrix = k_matrix/number_of_projectors

    # # my_graph_adjacency = torch.from_numpy(k_matrix)
    
    # G = gsp.graphs.Graph(W=my_graph_adjacency)
    # print(" Adjacency changed with the joint probabilities matrix")
    # ### -----------------------------------------------------------------------


    G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
    components = extract_components(G)
    print('the number of subgraphs is', len(components))
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            C, Gc, Call, Gall = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            C_list.append(C)
            Gc_list.append(Gc)
        number += 1
    return data.x.shape[1], len(set(np.array(data.y))), candidate, C_list, Gc_list

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def splits(data, num_classes, exp):
    if exp!='fixed':
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        else:
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data


def load_data(dataset, candidate, C_list, Gc_list, exp):
    if dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=dataset)
    else:
        dataset = Planetoid(root='./dataset', name=dataset)
    n_classes = len(set(np.array(dataset[0].y)))
    data = splits(dataset[0], n_classes, exp)
    train_mask = data.train_mask
    val_mask = data.val_mask
    labels = data.y
    features = data.x

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])

            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    print('the size of coarsen graph features:', coarsen_features.shape)

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge
