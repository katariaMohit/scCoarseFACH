import argparse
import pandas as pd
import os
import torch
import numpy as np
import GCN
import utils
import random
import networkx as nx
import torch_geometric
import time
from itertools import chain
import pyreadr as pyr
import h5py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean
import phate
import scprep
import meld
import sklearn
import tempfile
import scanpy as sc
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import GAT, GIN, Graph_Sage
import diff_pool_clustering
import seaborn as sns

import json

from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import SCAL.SCAL.APPNP.utils as SCAL_utils


def parse_arguments():
    parser = argparse.ArgumentParser(description="Graph Coarsening for Single Cell Data")
    parser.add_argument("--plotting", type=bool, required=False, default=False, help="If you want analysis of stored results. Program will terminate after that.")
    parser.add_argument("--dataset", type=str, required=False, default="all", help="Name of the dataset")
    parser.add_argument('--dataset_ratios', nargs='+', type=float, default=0.001, required=False, help='Our datasets are very large, this parameter specify the ratio of the datasets you want to use.(List is given as code will run for each dataset ration stored in list).')
    parser.add_argument("--meld_original", type=bool, required=False, default=False, help="Check Meld algorithm on original dataset.")
    parser.add_argument("--coarsen_whole_dataset", type=bool, required=False, default=False, help="Our algorithm is able to coarsen down whole dataset considering dataset in online setting.")
    parser.add_argument("--train_on_original_data", type=bool, required=False, default=False, help="Check GNN accuracy on original dataset.")
    parser.add_argument("--epochs", type=int, required=False, default=1000, help="Number of epochs.")
    parser.add_argument("--plot_graph", type=bool, required=False, default=False, help="Plot graphs")
    parser.add_argument("--gene_network", type=bool, required=False, default=False, help="gene_network")
    parser.add_argument("--sota_method", type=str, required=False, default="None", help="sota_method")
    parser.add_argument("--sota_ratio", type=float, required=False, default=0.5, help="sota method coarsening ratio")


    return parser.parse_args()

def hashed_values(data, no_of_hash,feature_size,function,projectors_distribution):
  print(data.dtype)
  
  if projectors_distribution == 'normal':
    Wl = torch.DoubleTensor(no_of_hash, feature_size).normal_(0,1)
  else:
    #uniform
    Wl = torch.DoubleTensor(no_of_hash, feature_size).uniform_(0,1)
  
  if data.dtype == torch.int64:
    data = data.to(torch.double)
    print(data.dtype)

  if function == 'L2-norm':
    Bin_values = torch.cdist(data, Wl, p = 2)
  elif function == 'L1-norm':
    Bin_values = torch.cdist(data, Wl, p = 1)
  else:
    #dot
    Bin_values = torch.matmul(data, Wl.T)
    
  return Bin_values

def val(model,data):
    data = data#.to(device)
    model.eval()
    # pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc

def partition(list_bin_width,Bin_values,no_of_hash):
    summary_dict = {}
    for bin_width in list_bin_width:
        bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)])#.to(device)
        temp = torch.floor((1/bin_width)*(Bin_values + bias))#.to(device)
        cluster, _ = torch.max(temp, dim = 1)
        dict_hash_indices = {}
        no_nodes = Bin_values.shape[0]
        for i in range(no_nodes):
            dict_hash_indices[i] = int(cluster[i]) #.to('cpu')
        summary_dict[bin_width] = dict_hash_indices 
    return summary_dict

def create_KNN_classification(celldataset,k,celllabel):
    A_KNN=kneighbors_graph(celldataset, k,include_self=True, mode="distance")
    A_KNN= np.array(A_KNN.toarray())
    return A_KNN

def coarsening_quality_measure(C_diag, dict_blabla, cell_labels, super_node_labels, num_classes):
    heatmap = np.zeros((num_classes,num_classes))

    for i in range(len(C_diag)):
        super_node_label = super_node_labels[i]
        print("super_node_label ", super_node_label)
        sub_nodes_label = cell_labels[dict_blabla[i]][0]
        print("sub_nodes_label ", sub_nodes_label, sub_nodes_label.shape)
        
        for sub_node_label in sub_nodes_label:
            heatmap[super_node_label][sub_node_label] = heatmap[super_node_label][sub_node_label] + 1
    
    print(heatmap)

def handle_gene_network(original_gene_data, gene_list, C_diag, dict_blabla):
    results = {}
    for i in range(len(C_diag)):
        print(C_diag[i])
        print(dict_blabla[i])
    print(len(C_diag))
    # A_array = create_KNN_classification(cell_data, k_value, concatenated_label)
    # np.fill_diagonal(A_array, 0)



def train_on_original_dataset(data, num_classes, feature_size, hidden_units, learning_rate, decay, epochs, num_testing_nodes=None):
  model = GCN.GCN_(feature_size, hidden_units, num_classes)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)

  if num_testing_nodes == 0:
    train_rate = 0.6
    val_rate = 0.2

    percls_trn = int(round(train_rate*len(data.y)/num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    
    data = utils.random_splits(data, num_classes, percls_trn, val_lb)
  else:
    data = utils.random_splits_large_data(data, num_classes, percls_trn, val_lb, num_testing_nodes)  

#   test_split_percent = 0.2
#   data = utils.split(data,num_classes,test_split_percent)
  
  if data.edge_attr == None:
    edge_weight = torch.ones(data.edge_index.size(1))
    data.edge_attr = edge_weight
    
  for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index,data.edge_attr.float())
    pred = out.argmax(1)
    criterion = torch.nn.NLLLoss()
    # print(data.train_mask)
    # print(out)
    # print(data.y)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].long()) 
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    best_val_acc = 0
    
    # val_acc = val(model,data)
    # if best_val_acc < val_acc:
    #     best_model = model #torch.save(model, 'full_best_model.pt')
    #     best_val_acc = val_acc
  
    if epoch % 20 == 0:
        print('Epoch ', epoch)
        # print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))
  
  model = model.to('cpu')
#   model = best_model #torch.load('full_best_model.pt')
  model.eval()
  data = data#.to(device)
  pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc = int(correct) / int(data.test_mask.sum())
  
  print('--------------------------')
  print('Accuracy on test data {:.3f}'.format(acc*100))

  return acc*100

def data_preprocessing_scRNA_1(data_file, label_file, data_name, percentage_to_use = 0.005):   
    data_df = pd.read_csv(data_file, delimiter=',')
    labels_df = pd.read_csv(label_file, delimiter=',')

    data_df = data_df.drop(data_df.columns[0], axis=1)
    # print(data_df)
    if data_name == 'amb':
        labels, label_indices = pd.factorize(labels_df['Class'])
    else:
        labels, label_indices = pd.factorize(labels_df['x'])

    node_labels = labels
    num_classes = torch.unique(torch.tensor(node_labels)).shape[0]

    return data_df, node_labels, num_classes

def data_preprocessing(data_name, input_file, data_directory, percentage_to_use = 0.005):   
    df = pd.read_csv(input_file).values
    concatenated_data = pd.DataFrame()
    concatenated_label = []

    counter = 0
    for file_name, label in df:
        if counter < 100:
            file_name = os.path.join(data_directory, file_name)
            # print("file number ", counter, " file name ",file_name)

            if 'LPS+IFNa' not in file_name:
                counter += 1

                if data_name == 'covid':
                    file_data = pd.read_csv(file_name, delimiter="\t")
                else:
                    file_data = pd.read_csv(file_name)
                    if data_name == 'nkcell':
                        file_data = file_data.drop('label', axis=1)

                file_data = file_data.sample(frac=percentage_to_use, random_state=42)

                # print(file_data.values.shape)
                if data_name == 'covid':
                    if label == 'healthy':
                        num_classes = 3
                        temp_label = [0] * file_data.values.shape[0]
                        # print(temp_label)
                        concatenated_label.append(temp_label)
                    elif label == 'ward':
                        temp_label = [1] * file_data.values.shape[0]
                        concatenated_label.append(temp_label)
                    else:
                        temp_label = [2] * file_data.values.shape[0]
                        concatenated_label.append(temp_label)
                elif data_name == 'nkcell':
                    num_classes = 2
                    if label == 0 or label == '0':
                        temp_label = [0] * file_data.values.shape[0]
                        # print(temp_label)
                        concatenated_label.append(temp_label)
                    else:
                        temp_label = [1] * file_data.values.shape[0]
                        concatenated_label.append(temp_label)
                else:
                    num_classes = 2
                    if label == 'Control':
                        temp_label = [0] * file_data.values.shape[0]
                        # print(temp_label)
                        concatenated_label.append(temp_label)
                    else:
                        temp_label = [1] * file_data.values.shape[0]
                        concatenated_label.append(temp_label)

                concatenated_data = pd.concat([concatenated_data, file_data], axis=0)

    concatenated_label =  list(chain.from_iterable(concatenated_label)) 
    concatenated_label = np.array(concatenated_label)

    # print(concatenated_data.head())  # You can display the concatenated data or further process it
    # print(concatenated_data.values.shape)
    # print("label information ")
    # print(concatenated_label)

    return concatenated_data, concatenated_label, num_classes

def check_intra_node_similarity(partition_matrix,labels):
    max_counts = 0
    super_nodes_labels = []
    for j in range(partition_matrix.shape[0]):
        subnodes_in_supernode = np.where(partition_matrix[j] == 1)[0]
        subnode_labels = labels[subnodes_in_supernode]
        counts = np.bincount(subnode_labels)
        
        ## max ocuuring label, no_of_time it occurs
        max_index, max_element = max(enumerate(counts), key=lambda x: x[1])

        max_counts += max_element
        super_nodes_labels.append(max_index)
    
    return max_counts, super_nodes_labels

def get_intra_node_error(partition_matrix, super_nodes_labels,labels):
    trace_different=0
    # print(torch.nonzero(partition_matrix[0]))
    # print(labels)
    super_nodes_labels = np.array(super_nodes_labels)
    labels = np.array(labels)
    for i in range(partition_matrix.shape[0]):
        # print(labels[torch.nonzero(partition_matrix[0])])
        # print(super_nodes_labels[i])
        trace_different += np.sum(np.abs(labels[torch.nonzero(partition_matrix[0])] - super_nodes_labels[i]))
    
    return trace_different


def plotting_function():
    import json

    # Replace 'your_file.json' with the path to your JSON file
    file_path = 'results.json'

    # Read the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Create a new dictionary with split keys
    new_data = {'covid': {'0.001': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                        '0.002': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                        '0.003': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                        '0.004': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                        '0.005': {'acc_list': [], 'err_list': [], 'cor_ratio': []}},
                
                
                'nkcell': {'0.001': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                            '0.002': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                            '0.003': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                            '0.004': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                            '0.005': {'acc_list': [], 'err_list': [], 'cor_ratio': []}},
                
                
                'preeclampsia': {'0.001': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                                '0.002': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                                '0.003': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                                '0.004': {'acc_list': [], 'err_list': [], 'cor_ratio': []},
                                '0.005': {'acc_list': [], 'err_list': [], 'cor_ratio': []}}
                }

    for key, value in data.items():
        [data_name, _, data_ratio, _, coarsening_ratio] = key.split('_')
        
        new_data[data_name][data_ratio]['cor_ratio'].append(float(coarsening_ratio) * 100)
        new_data[data_name][data_ratio]['acc_list'].append(float(value['accuracy']) * 100)
        new_data[data_name][data_ratio]['err_list'].append(float(value['error']))


    # print(new_data)

    import matplotlib.pyplot as plt


    for data_name in new_data.keys():
        # Iterate over each data_ratio
        for data_ratio in new_data[data_name].keys():
            # Extract the cor_ratio and corresponding lists
            cor_ratio = new_data[data_name][data_ratio]['cor_ratio']
            acc_list = new_data[data_name][data_ratio]['acc_list']
            err_list = new_data[data_name][data_ratio]['err_list']

            # Create a new figure for accuracy vs. coarsening ratio
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(cor_ratio, acc_list,marker='o')
            plt.title(f'Accuracy vs. Coarsening Ratio for {data_name} ({data_ratio})')
            plt.xlabel('Coarsening Ratio')
            plt.ylabel('Accuracy')

            # Create a new figure for error vs. coarsening ratio
            plt.subplot(1, 2, 2)
            plt.plot(cor_ratio, err_list, marker='o')
            plt.title(f'Error vs. Coarsening Ratio for {data_name} ({data_ratio})')
            plt.xlabel('Coarsening Ratio')
            plt.ylabel('Error')

            # Show or save the figures (adjust as needed)
            plt.tight_layout()
            file_name = data_name + '_' + data_ratio + '.jpg'
            plt.savefig(file_name)
            plt.show()

def plotting_function_1():
    import json

    # Replace 'your_file.json' with the path to your JSON file
    file_path = 'results_including_meld_covid.json'

    # Read the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Create a new dictionary with split keys
    new_data = {'covid': {'0.001': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [], 'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                        '0.002': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [], 'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                        '0.003': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                        '0.004': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                        '0.005': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]}},
                
                
                'nkcell': {'0.001': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                            '0.002': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                            '0.003': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                            '0.004': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                            '0.005': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]}},
                
                
                'preeclampsia': {'0.001': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                                '0.002': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                                '0.003': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                                '0.004': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]},
                                '0.005': {'intra_node_accuracy_list': [], 'err_list': [], 'cor_ratio': [],  'meld_accuracy_after_ugc_coarsened_graph_list':[], 'meld_algo_accuracy_for_this_data_list':[],'gcn_accuracy_corsened_graph_list':[]}}
                }

    for key, value in data.items():
        [data_name, _, data_ratio, _, coarsening_ratio] = key.split('_')
        
        new_data[data_name][data_ratio]['cor_ratio'].append(float(coarsening_ratio) * 100)
        new_data[data_name][data_ratio]['intra_node_accuracy_list'].append(float(value['intra_node_accuracy']) * 100)
        new_data[data_name][data_ratio]['err_list'].append(float(value['error']))
        new_data[data_name][data_ratio]['meld_accuracy_after_ugc_coarsened_graph_list'].append(float(value['meld_accuracy_after_ugc_coarsened_graph']) * 100)
        new_data[data_name][data_ratio]['meld_algo_accuracy_for_this_data_list'].append(float(value['meld_algo_accuracy_for_this_data']) * 100)
        new_data[data_name][data_ratio]['gcn_accuracy_corsened_graph_list'].append(float(value['gcn_accuracy_corsened_graph']) * 100)


    # print(new_data)

    import matplotlib.pyplot as plt


    for data_name in new_data.keys():
        # Iterate over each data_ratio
        for data_ratio in new_data[data_name].keys():
            # Extract the cor_ratio and corresponding lists
            cor_ratio = new_data[data_name][data_ratio]['cor_ratio']
            acc_list = new_data[data_name][data_ratio]['intra_node_accuracy_list']
            err_list = new_data[data_name][data_ratio]['err_list']
            meld_acc_after_ugc = new_data[data_name][data_ratio]['meld_accuracy_after_ugc_coarsened_graph_list']
            meld_algo_acc = new_data[data_name][data_ratio]['meld_algo_accuracy_for_this_data_list']
            gcn_accuracy_corsened = new_data[data_name][data_ratio]['gcn_accuracy_corsened_graph_list']

            # Create a new figure for accuracy vs. coarsening ratio
            plt.figure(figsize=(25, 5))
            plt.subplot(1, 5, 1)
            plt.plot(cor_ratio, acc_list,marker='o')
            plt.title(f'Accuracy vs. Coarsening Ratio for {data_name} ({data_ratio})')
            plt.xlabel('Coarsening Ratio')
            plt.ylabel('Accuracy')

            # Create a new figure for error vs. coarsening ratio
            plt.subplot(1, 5, 2)
            plt.plot(cor_ratio, err_list, marker='o')
            plt.title(f'Error vs. Coarsening Ratio for {data_name} ({data_ratio})')
            plt.xlabel('Coarsening Ratio')
            plt.ylabel('Error')
            

            print("data name", data_name," data ratio ",data_ratio," cor_ratio ",cor_ratio, " meld_acc ",meld_acc_after_ugc, "intra node similarity ",acc_list)
            
            # Create a new figure for error vs. coarsening ratio
            plt.subplot(1, 5, 3)
            plt.plot(cor_ratio, meld_acc_after_ugc, marker='o')
            plt.title(f'Meld acc with UGC vs. Coarsening Ratio for {data_name} ({data_ratio})')
            plt.xlabel('Coarsening Ratio')
            plt.ylabel('Acc')

            # Create a new figure for error vs. coarsening ratio
            plt.subplot(1, 5, 4)
            plt.plot(cor_ratio, meld_algo_acc, marker='o')
            plt.title(f'Meld Algo acc vs. Coarsening Ratio for {data_name} ({data_ratio})')
            plt.xlabel('Coarsening Ratio')
            plt.ylabel('Acc')

            # Create a new figure for error vs. coarsening ratio
            plt.subplot(1, 5, 5)
            plt.plot(cor_ratio, gcn_accuracy_corsened, marker='o')
            plt.title(f'GCN accuracy vs. Coarsening Ratio for {data_name} ({data_ratio})')
            plt.xlabel('Coarsening Ratio')
            plt.ylabel('Acc')

            # Show or save the figures (adjust as needed)
            plt.tight_layout()
            file_name = data_name + '_' + data_ratio + '_exp2_' + '.jpg'
            plt.savefig(file_name)
            plt.show()

    

def normalize_densities(sample_densities):
    """
    Takes a 2-d array of sample densities from the same replicate and
    normalizes the row-sum to 1.
    """
    if isinstance(sample_densities, pd.DataFrame):
        index, columns = sample_densities.index, sample_densities.columns

    norm_densities = sklearn.preprocessing.normalize(sample_densities, norm="l1")

    if isinstance(sample_densities, pd.DataFrame):
        norm_densities = pd.DataFrame(norm_densities, index=index, columns=columns)
        
    return norm_densities

def plot_meld(file_name, num_classes, data_phate, sample_densities, labels):
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    # Create a dictionary to store the class index and its corresponding color
    print(num_classes)
    sample_cmap = {i: mcolors.rgb2hex(cmap(i)[:3]) for i in range(num_classes)}
    print(sample_cmap)

    # max_indices = []
    # ### Plot original data
    # for row in sample_densities:
    #     max_index = np.argmax(row)
    #     print(max_index, " row ",row)
    #     max_indices.append(max_index)

    # # Convert the list to a 1D numpy array
    # max_indices = np.array(max_indices)

    # print(file_name)
    # scprep.plot.scatter2d(data_phate, c=max_indices, cmap=sample_cmap, legend=False,
    #                     legend_anchor=(1,1), figsize=(6,5), s=10, label_prefix=None, ticks=False, filename=file_name)
    
    # print(file_name)

    # scprep.plot.scatter2d(data_phate, c=labels, cmap=sample_cmap, legend=False,
    #                     legend_anchor=(1,1), figsize=(6,5), s=10, label_prefix=None, ticks=False, filename=file_name)

    # plt.savefig(file_name)
    # plt.show()

    # fig, axes = plt.subplots(1,num_classes, figsize=(11,6))

    # for i, ax in enumerate(axes.flatten()):
    #     # density = sample_densities.iloc[:,i]
    #     density = sample_densities[i]
    #     print(density)
    #     scprep.plot.scatter2d(data_phate, c=density,
    #                         title=i,
    #                         vmin=0, 
    #                         ticks=False, ax=ax)

    # fig.tight_layout()

    

def run_meld(file_name, concatenated_data, labels, original_phate=None):
    phate_op = phate.PHATE(n_jobs=-1)
    data_phate = phate_op.fit_transform(concatenated_data)

    # sample_cmap = {
    #     0: '#fb6a4a',  # original color for class 0
    #     1: '#08519c',  # original color for class 1
    #     2: '#34a853',  # green
    #     3: '#ffeb3b',  # yellow
    #     4: '#ff9800',  # orange
    #     5: '#e91e63',  # pink
    #     6: '#9c27b0',  # purple
    #     7: '#00bcd4',  # cyan
    #     8: '#8bc34a',  # light green
    #     9: '#cddc39',  # lime
    #     10: '#795548',  # brown
    #     11: '#ff5722',  # deep orange
    #     12: '#607d8b',  # blue grey
    #     13: '#000000'   # black
    # }

    num_classes = torch.unique(torch.tensor(labels)).shape[0]
    # cmap = plt.cm.get_cmap('tab20', num_classes)
    
    # # Create a dictionary to store the class index and its corresponding color
    # sample_cmap = {i: mcolors.rgb2hex(cmap(i)[:3]) for i in range(num_classes)}

    # scprep.plot.scatter2d(data_phate, c=labels, cmap=sample_cmap, 
    #                     legend_anchor=(1,1), figsize=(6,5), s=10, label_prefix='PHATE', ticks=False)

    # beta is the amount of smoothing to do for density estimation
    # knn is the number of neighbors used to set the kernel bandwidth
    meld_op = meld.MELD(beta=67, knn=7)
    sample_densities = meld_op.fit_transform(concatenated_data, sample_labels=labels)
    # fig, axes = plt.subplots(1,2, figsize=(11,6))

    # for i, ax in enumerate(axes.flatten()):
    #     density = sample_densities.iloc[:,i]
    #     scprep.plot.scatter2d(data_phate, c=density,
    #                         title=density.name,
    #                         vmin=0, 
    #                         ticks=False, ax=ax)
    
    # plt.savefig(file_name)
    # fig.tight_layout()

    sample_likelihoods = normalize_densities(sample_densities)
    if original_phate != None:
        data_phate = original_phate

    orignial_meld_pred = [1 if prob[1] > prob[0] else 0 for prob in sample_likelihoods.values]

    # Calculate accuracy
    accuracy = accuracy_score(labels, orignial_meld_pred)
    print("Accuracy of meld score is ",accuracy)
    return orignial_meld_pred, accuracy, data_phate, sample_likelihoods

def meld_analysis(original_sample_likelihoods, broadcasted_meld_likelihoods, original_nodes_labels, super_nodes_labels):
    abs_diff = []
    print(original_sample_likelihoods)
    print(broadcasted_meld_likelihoods)

    for i in range(np.shape(original_sample_likelihoods)[0]):
        print(original_sample_likelihoods[i])
        print(broadcasted_meld_likelihoods[i])
        abs_diff.append(np.abs(original_sample_likelihoods[i] - broadcasted_meld_likelihoods[i]))
    
    abs_diff = np.array(abs_diff)

    print(original_nodes_labels)
    print(abs_diff.T[0])

    current_label_index = np.where(original_nodes_labels == 0)[0]
    current_label_abs = abs_diff[current_label_index]

    plt.plot(current_label_abs.T[0])
    # plt.plot(current_label_abs.T[1])
    # plt.plot(current_label_abs.T[2])
    # plt.plot(current_label_abs.T[3])
    plt.show()


    current_label_index = np.where(original_nodes_labels == 1)[0]
    current_label_abs = abs_diff[current_label_index]

    plt.plot(current_label_abs.T[1])
    # plt.plot(current_label_abs.T[1])
    # plt.plot(current_label_abs.T[2])
    # plt.plot(current_label_abs.T[3])
    plt.show()


    current_label_index = np.where(original_nodes_labels == 2)[0]
    current_label_abs = abs_diff[current_label_index]

    plt.plot(current_label_abs.T[2])
    # plt.plot(current_label_abs.T[1])
    # plt.plot(current_label_abs.T[2])
    # plt.plot(current_label_abs.T[3])
    plt.show()


    current_label_index = np.where(original_nodes_labels == 3)[0]
    current_label_abs = abs_diff[current_label_index]

    plt.plot(current_label_abs.T[3])
    # plt.plot(current_label_abs.T[1])
    # plt.plot(current_label_abs.T[2])
    # plt.plot(current_label_abs.T[3])
    plt.show()

    d = pd.DataFrame({'Labels': original_nodes_labels, 'Errors': abs_diff.T[0]})
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Labels', y='Errors', data=d, palette='viridis')
    plt.title('Violin Plot of Errors by Class')
    plt.xlabel('Class Labels')
    plt.ylabel('Error Values')
    plt.savefig("results/abs_meld_error_0")
    plt.show()

    d = pd.DataFrame({'Labels': original_nodes_labels, 'Errors': abs_diff.T[1]})
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Labels', y='Errors', data=d, palette='viridis')
    plt.title('Violin Plot of Errors by Class')
    plt.xlabel('Class Labels')
    plt.ylabel('Error Values')
    plt.savefig("results/abs_meld_error_1")
    plt.show()

    d = pd.DataFrame({'Labels': original_nodes_labels, 'Errors': abs_diff.T[2]})
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Labels', y='Errors', data=d, palette='viridis')
    plt.title('Violin Plot of Errors by Class')
    plt.xlabel('Class Labels')
    plt.ylabel('Error Values')
    plt.savefig("results/abs_meld_error_2")
    plt.show()


    d = pd.DataFrame({'Labels': original_nodes_labels, 'Errors': abs_diff.T[3]})
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Labels', y='Errors', data=d, palette='viridis')
    plt.title('Violin Plot of Errors by Class')
    plt.xlabel('Class Labels')
    plt.ylabel('Error Values')
    plt.savefig("results/abs_meld_error_3")
    plt.show()

    # exit(1)

    return


def get_meld_accuracy(partition_matrix, orignial_meld_pred, original_data, super_nodes_labels, file_name, original_data_phate, num_classes, original_sample_likelihoods, original_nodes_labels):
    super_node_features = []
    for j in range(partition_matrix.shape[0]):
        subnodes_in_supernode = np.where(partition_matrix[j] == 1)[0]
        # print(" super node ", j, " has ",subnodes_in_supernode," as subnodes.")
        # for subnode in subnodes_in_supernode:
        #     print("subnode number ",subnode, " feature of this subnode ",original_data[subnode])
        subnode_feature = np.sum(original_data[subnodes_in_supernode],axis=0)/len(subnodes_in_supernode)
        # print(" original feature vector ",original_data[subnodes_in_supernode])
        # print(" feature vector of supernode its shape is ",subnode_feature.shape,"  ",subnode_feature)
        super_node_features.append(subnode_feature)
   
    time_coarsend_meld0 = time.time()
    meld_labels_coarsen_graph, _, _, coarsen_sample_likelihoods = run_meld(file_name, np.array(super_node_features), super_nodes_labels)
    time_coarsend_meld1 = time.time()
    

    print("corsened data meld ",time_coarsend_meld1-time_coarsend_meld0)

    broadcasted_meld_likelihoods = {}#np.zeros((partition_matrix.shape[1],coarsen_sample_likelihoods.shape[1]))
    broadcasted_labels = np.zeros(partition_matrix.shape[1])

    for j in range(partition_matrix.shape[0]):
        subnodes_in_supernode = np.where(partition_matrix[j] == 1)[0]
        # print("subnodes_in_supernode ", subnodes_in_supernode)
        for subnode in subnodes_in_supernode:
            # print(coarsen_sample_likelihoods.values.T.shape, coarsen_sample_likelihoods.values.T)
            broadcasted_meld_likelihoods[subnode] = coarsen_sample_likelihoods.values[j]
            broadcasted_labels[subnode] = super_nodes_labels[j]

    
    ###
    broadcasted_meld_likelihoods_values = list(broadcasted_meld_likelihoods.values())
    broadcasted_meld_likelihoods_matrix = np.array(broadcasted_meld_likelihoods_values)
    meld_analysis(original_sample_likelihoods, broadcasted_meld_likelihoods_matrix, original_nodes_labels, super_nodes_labels)
    ###

    
    ####  plotting after broadcasted to original space from coarsened space
    # print(broadcasted_meld_likelihoods.values())
    # exit(1)
    plot_meld(file_name, num_classes, original_data_phate, broadcasted_meld_likelihoods.values(), broadcasted_labels)
    

    ############  this should not be correct it divides the meld score into two parts only no sense of accuracy here
    meld_pred_coarsen_labels = np.zeros(partition_matrix.shape[1])
    for j in range(partition_matrix.shape[0]):
        subnodes_in_supernode = np.where(partition_matrix[j] == 1)[0]
        # print("subnodes_in_supernode ", subnodes_in_supernode)
        for subnode in subnodes_in_supernode:
            meld_pred_coarsen_labels[subnode] = meld_labels_coarsen_graph[j]
    ############

    # print(np.array(orignial_meld_pred))
    # print(meld_pred_coarsen_labels)

    print(" orignial_meld_pred shape ",len(orignial_meld_pred))
    print(" meld_pred_coarsen_labels ",meld_pred_coarsen_labels.shape)

    accuracy = accuracy_score(np.array(orignial_meld_pred), meld_pred_coarsen_labels)
    print(" meld accuracy using the coarsened graph ",accuracy)

    return accuracy, time_coarsend_meld1-time_coarsend_meld0

def UGC_in_chunks(current_summary_dict, bin_width, data_in_one_chunk,chunk_in, weight_projectors, bias_projectors, data):
    # print(current_summary_dict) 
    total_entries = data.shape[0]
    entries_to_consider = (int)(total_entries*data_in_one_chunk)

    no_of_hash = 1000
    feature_size = data.shape[1]
    data_seen_till_now = chunk_in/total_entries*100 
    print("total_entries ",total_entries," entries_to_consider ",entries_to_consider," percent data_seen_till_now ",data_seen_till_now)
    print("chunk start ",chunk_in," chunk end ",chunk_in + entries_to_consider)

    if current_summary_dict == {}:
        print(" intilizing the projectors ")
        weight_projectors = torch.DoubleTensor(no_of_hash, feature_size).normal_(0,1)
        bias_projectors = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)])
    
    if chunk_in + entries_to_consider > data.shape[0]:
        print(" should be printed only once at the end of the iteration.",chunk_in + entries_to_consider,total_entries)
        entries_to_consider = data.shape[0] - chunk_in


    current_Bin_values = torch.matmul(data[chunk_in:chunk_in+entries_to_consider], weight_projectors.T)
    temp = torch.floor((1/bin_width)*(current_Bin_values + bias_projectors))
    cluster, _ = torch.max(temp, dim = 1)

    for i in range(entries_to_consider):
        # print(" chunk_in + i ",chunk_in + i,"  ",int(cluster[i]))
        current_summary_dict[chunk_in + i] = int(cluster[i])

    if chunk_in + entries_to_consider >= total_entries:
        return current_summary_dict
    
    return UGC_in_chunks(current_summary_dict, bin_width, data_in_one_chunk,chunk_in + entries_to_consider, weight_projectors, bias_projectors, data)


def get_partition_matrix_and_supernode_features(results, data, labels):
    # print("get_partition_matrix_and_supernode_features")
    # for label in labels:
    #     print(label)
    values = results.values()
    unique_values = set(values)
    rr = 1 - len(unique_values)/len(values)
    print(f'Graph reduced by: {rr*100} percent.\nWe now have {len(unique_values)} supernode, starting nodes were: {len(values)}')
    dict_blabla ={}
    C_diag = torch.zeros(len(unique_values))#, device= device)
    help_count = 0
    
    # i thinnk this can be improved
    # does this have a time complexity if O(N*v) ? i.e for each unique value searching each node hash value
    for v in unique_values:
        C_diag[help_count],dict_blabla[help_count] = utils.get_key(v, results)
        help_count += 1

    ###----------------------------In the matrix form we need to change this to list or dict format
    # P_hat is bool 2D array which represent nodes contained in supernodes 
    partition_matrix = torch.zeros((data.shape[0], len(unique_values)))#, device= device)
    for x in dict_blabla:
        if len(dict_blabla[x]) == 0:
            print("zero element in this supernode",x)
        for y in dict_blabla[x]:
            partition_matrix[y,x] = 1

    print("partition_matrix calculated")
    super_nodes_labels = []
    super_node_features = []

    for j in range(partition_matrix.shape[0]):
        subnodes_in_supernode = np.where(partition_matrix[j] == 1)[0]
        subnode_labels = labels[subnodes_in_supernode]
        counts = np.bincount(subnode_labels)       
        ## max ocuuring label, no_of_time it occurs
        max_index, max_element = max(enumerate(counts), key=lambda x: x[1])

        # print(subnode_labels,counts,max_index,max_element)
        super_nodes_labels.append(max_index)
        # print(np.sum(data[subnodes_in_supernode], axis=0)/len(subnodes_in_supernode))
        super_node_features.append(np.sum(data[subnodes_in_supernode], axis=0)/len(subnodes_in_supernode))
    ###----------------------------

    super_nodes_labels = []
    super_node_features = []

    for j in range(partition_matrix.shape[0]):
        subnodes_in_supernode = np.where(partition_matrix[j] == 1)[0]
        subnode_labels = labels[subnodes_in_supernode]
        counts = np.bincount(subnode_labels)       
        ## max ocuuring label, no_of_time it occurs
        max_index, max_element = max(enumerate(counts), key=lambda x: x[1])

        # print(subnode_labels,counts,max_index,max_element)
        super_nodes_labels.append(max_index)
        # print(np.sum(data[subnodes_in_supernode], axis=0)/len(subnodes_in_supernode))
        super_node_features.append(np.sum(data[subnodes_in_supernode], axis=0)/len(subnodes_in_supernode))

    return super_nodes_labels, super_node_features, partition_matrix


def visulize_data(data, labels, coarsened_Data = False):
    print("visulize_data")
    if coarsened_Data == False:
        sampling_ratio = 0.01

        # Set a random seed for reproducibility
        random_seed = 42

        # Reset the index for both data and labels DataFrames
        concatenated_data = concatenated_data.reset_index(drop=True)
        concatenated_label = concatenated_label.reset_index(drop=True)

        # Randomly drop entries based on the sampling ratio using the same random seed
        sampled_data = concatenated_data.sample(frac=sampling_ratio, random_state=random_seed)

        print(sampled_data.index)
        sampled_labels = concatenated_label.loc[sampled_data.index]


        # Sample data: You should replace this with your own feature matrix and labels
        X = sampled_data.values  # Replace with your feature matrix
        y = sampled_labels.values

        print(X.shape)
        print(y.shape)

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
    else:

        # Sample data: You should replace this with your own feature matrix and labels
        X = data 
        y = labels
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

    # Create a new DataFrame for the t-SNE results
    tsne_df = pd.DataFrame(data=X_tsne, columns=['Dimension_1', 'Dimension_2'])
    tsne_df['Label'] = y

    # Visualize the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_df['Dimension_1'], tsne_df['Dimension_2'], c=tsne_df['Label'], cmap='viridis')
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def balanced_sampling(data, entry_per_class):

    from collections import defaultdict
    indices_by_class = defaultdict(list)
    for index, value in enumerate(list(data)):
        indices_by_class[value].append(index)

    # print(indices_by_class.values())

    min_class_size = min(len(indices) for indices in indices_by_class.values())
    
    if min_class_size > entry_per_class:
        min_class_size = entry_per_class
    else:
        print("mentioned dataset can't be fetched in a balanced setting please check")
        exit(1)

    sampled_indices = []
    for indices in indices_by_class.values():
        sampled_indices.extend(random.sample(indices, min_class_size))

    return sampled_indices

def sample_nodes_from_supernodes(partition_matrix, number_of_subnodes, super_node_labels):
    sampled_labels = []
    sampled_labels_index = []
    i = 0
    keys = list(partition_matrix.keys())
    # print(keys)

    # print(partition_matrix.values())
    # exit(1)

    while i < number_of_subnodes:
        super_node_number = random.randint(0, len(keys)-1)
        # print(len(keys), super_node_number)
        # print(keys[super_node_number])
        subnodes_in_supernode = partition_matrix[keys[super_node_number]]
        sampled_sub_node_number = subnodes_in_supernode[random.randint(0, len(subnodes_in_supernode)-1)]
        # print("len(subnodes_in_supernode) ",len(subnodes_in_supernode), " sampled_sub_node_number ", sampled_sub_node_number)
        # print(sampled_labels_index)
        if sampled_sub_node_number not in sampled_labels_index:
            i = i + 1
            sampled_labels_index.append(sampled_sub_node_number)
            sampled_labels.append(super_node_labels[super_node_number])

    return sampled_labels_index, sampled_labels


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device ", device)
    ####_____________________ Helper functions for plotting graphs_______________
    if args.plotting != False:
        ## choose plotting function
        # 1
        # plotting_function()
        
        # 2
        plotting_function_1()
        
        print("Done with Result analysis. Terminating......")
        exit()
    ####_________________________________________________________________________


    start_time = time.time()

    ### used for dumping the results
    information_dict = {}

    dataset_name = args.dataset
    print("You choose ",dataset_name, " dataset.")

    ## Datasets path initilization as long as you maintain maintain datasets in data\dataset_name Dataset we don't need to change these lines
    data_path = {}
    data_repo_path = {}
    label_path = {}
    
    if dataset_name == 'covid': 
        data_list = ['covid']
        data_repo_path['covid'] = r"data\Covid Dataset"
        data_path['covid'] = r"data\Covid Dataset\covid_dataset.csv"

    elif dataset_name == 'nkcell': 
        data_list = ['nkcell']
        data_repo_path['nkcell'] = r"data\NK Cell Dataset"
        data_path['nkcell'] = r"data\NK Cell Dataset\NKcell_dataset.csv"

    elif dataset_name == 'preeclampsia': 
        data_list = ['preeclampsia']
        data_repo_path['preeclampsia'] = r"data\Preeclampsia Dataset"
        data_path['preeclampsia'] = r"data\Preeclampsia Dataset\Han-FCS_file_list.csv"

    elif dataset_name == 'xin': 
        data_list = ['xin']
        data_repo_path['xin'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin"
        data_path['xin'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin\Filtered_Xin_HumanPancreas_data.csv"
        label_path['xin'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin\Labels.csv"

    elif dataset_name == 'baron_mouse': 
        data_list = ['baron_mouse']
        data_repo_path['baron_mouse'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse"
        data_path['baron_mouse'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse\Filtered_MousePancreas_data.csv"
        label_path['baron_mouse'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse\Labels.csv"

    elif dataset_name == 'baron_human': 
        data_list = ['baron_human']
        data_repo_path['baron_human'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human"
        data_path['baron_human'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human\Filtered_Baron_HumanPancreas_data.csv"
        label_path['baron_human'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human\Labels.csv"

    elif dataset_name == 'muraro': 
        data_list = ['muraro']
        data_repo_path['muraro'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro"
        data_path['muraro'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro\Filtered_Muraro_HumanPancreas_data.csv"
        label_path['muraro'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro\Labels.csv"

    elif dataset_name == 'segerstolpe': 
        data_list = ['segerstolpe']
        data_repo_path['segerstolpe'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe"
        data_path['segerstolpe'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe\Filtered_Segerstolpe_HumanPancreas_data.csv"
        label_path['segerstolpe'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe\Labels.csv"

    elif dataset_name == 'amb': 
        data_list = ['amb']
        data_repo_path['amb'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB"
        data_path['amb'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB\Filtered_mouse_allen_brain_data.csv"
        label_path['amb'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB\Labels.csv"

    elif dataset_name == 'tm': 
        data_list = ['tm']
        data_repo_path['tm'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM"
        data_path['tm'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM\Filtered_TM_data.csv"
        label_path['tm'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM\Labels.csv"

    elif dataset_name == 'zheng': 
        data_list = ['zheng']
        data_repo_path['zheng'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng"
        data_path['zheng'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng\Filtered_68K_PBMC_data.csv"
        label_path['zheng'] = r"scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng\Labels.csv"

    ###1 
    elif dataset_name == 'baron_5000':
        data_list = ['baron_5000']
        data_repo_path['baron_5000'] = r"new_data\1"
        data_path['baron_5000'] = r"new_data\1\baron-5000.rds"

    elif dataset_name == 'darmanis':
        data_list = ['darmanis']
        data_repo_path['darmanis'] = r"new_data\1"
        data_path['darmanis'] = r"new_data\1\darmanis.rds"

    elif dataset_name == 'deng':
        data_list = ['deng']
        data_repo_path['deng'] = r"new_data\1"
        data_path['deng'] = r"new_data\1\deng.rds"
    
    ####2
    elif dataset_name == 'mECS':
        data_list = ['mECS']
        data_repo_path['mECS'] = r"new_data\2"
        data_path['mECS'] = r"new_data\2\Test_1_mECS.RData"

    elif dataset_name == 'Kolod':
        data_list = ['Kolod']
        data_repo_path['Kolod'] = r"new_data\2"
        data_path['Kolod'] = r"new_data\2\Test_2_Kolod.RData"

    ####4
    elif dataset_name == 'PBMC':
        data_list = ['PBMC']
        data_repo_path['PBMC'] = r"new_data\4"
        data_path['PBMC'] = r"new_data\4\PBMC_68k.h5"

    else:
        print("Dataset ",dataset_name, " is currently not supported. Exiting ......")
        exit(1)


    if dataset_name not in ['covid','nkcell','preeclampsia'] or args.coarsen_whole_dataset == True:
        dataset_ratios = [1.0]
    else:
        # dataset_ratios = args.dataset_ratios
        if dataset_name == 'nkcell':
            dataset_ratios = [0.07]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]
        elif dataset_name == 'covid':
            dataset_ratios = [0.01]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]
        else:
            dataset_ratios = [0.006]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]    
        # dataset_ratios = [0.001, 0.002, 0.003, 0.004, 0.005]#, 0.01]
        # dataset_ratios = [1.0]#, 0.02,0.03, 0.04]
    
    
    results = {}
    # making sure plots & clusters are reproducible
    np.random.seed(42)

    for data_name in data_list:
        for dataset_ratio in dataset_ratios:           
            dir_path = data_repo_path[data_name]
            file_path = data_path[data_name]
            # print(dir_path)
            # print(file_path)     
            if dataset_name in ['covid','nkcell','preeclampsia']:
                concatenated_data, concatenated_label, num_classes = data_preprocessing(data_name, file_path, dir_path, dataset_ratio)
                unique, counts = np.unique(concatenated_label, return_counts=True)
                print(unique, counts)
                cell_data = concatenated_data.values
            
            elif dataset_name in ['baron_5000', 'darmanis', 'deng', 'mECS', 'Kolod']:
                result = pyr.read_r(file_path) # also works for RData
                print(result)
                data=pd.DataFrame(result)
                print(data)
                exit(1)

            elif dataset_name in ['PBMC']:
                with h5py.File(file_path, "r") as h5f:
                    data = h5f["dataset"][:]
                    print(data)
                exit(1)

            else:
                label_path = label_path[data_name]
                concatenated_data, concatenated_label, num_classes = data_preprocessing_scRNA_1(file_path, label_path, data_name)
                cell_data = concatenated_data.values

            if args.gene_network == True:
                gene_list = list(concatenated_data.columns)
                original_gene_data = cell_data.T
                cell_data = cell_data.T

            ################## Meld accuracy on original data
            if args.meld_original != False:
                file_name = 'results/Meld_original_' + (str)(data_name)            
                time_meld0 = time.time()
                orignial_meld_pred, meld_accuracy_original_graph, original_data_phate, sample_likelihoods = run_meld(file_name,concatenated_data, concatenated_label)
                time_meld1 = time.time()
                print("full meld time ",time_meld1 - time_meld0)
                plot_meld(file_name, num_classes, original_data_phate, sample_likelihoods.values, concatenated_label)
                
                information_dict['meld_algo_accuracy_for_this_data'] = meld_accuracy_original_graph
                information_dict['whole_meld_time'] = time_meld1 - time_meld0
            ###########################

            #############
            # if dataset_name in ['covid','nkcell','preeclampsia']:
            #     testing_nodes = balanced_sampling(concatenated_label, 1000)
            #     # print(testing_nodes)
            #     testing_data = cell_data[testing_nodes]
            #     testing_label = concatenated_label[testing_nodes]
                
            #     # print(cell_data.shape)
            #     # print(concatenated_label.shape)

            #     mask = np.ones(concatenated_label.shape[0], dtype=bool)
            #     mask[testing_nodes] = False    

            #     cell_data = cell_data[mask]
            #     concatenated_label = concatenated_label[mask]
            #     # print(cell_data.shape)
            #     # print(concatenated_label.shape)
            #     # print(testing_data.shape)
            #     # print(testing_label.shape)
            #     # exit(1)

            # else:
            testing_nodes = []
            #############

            ################## Processing the whole data
            ### Need to improve this by making sure that we store partition matrix in the form of list not matrix
            if args.coarsen_whole_dataset != False:
                testing_nodes = balanced_sampling(concatenated_label, 1000)
                # print(testing_nodes)
                testing_data = cell_data[testing_nodes]
                testing_label = concatenated_label[testing_nodes]
                
                # print(cell_data.shape)
                # print(concatenated_label.shape)

                mask = np.ones(concatenated_label.shape[0], dtype=bool)
                mask[testing_nodes] = False    

                # cell_data = cell_data[mask]
                # concatenated_label = concatenated_label[mask]
                # print(cell_data.shape)
                # print(concatenated_label.shape)
                # print(testing_data.shape)
                # print(testing_label.shape)
                # exit(1)

                current_summary_dict = {}
                data_in_one_chunk = 0.005
                chunk_in = 0
                weight_projectors = None
                bias_projectors = None
                if data_name == 'covid':
                    bin_width_1 = 50 
                elif data_name == 'nkcell':
                    bin_width_1 = 0.0008
                else:
                    bin_width_1 = 0.001

                whole_graph_start_time = time.time()
                whole_UGC_results = UGC_in_chunks(current_summary_dict, bin_width_1, data_in_one_chunk,chunk_in, weight_projectors, bias_projectors, torch.tensor(cell_data))
                print("done with whole data partition ",len(whole_UGC_results.keys()))
                ### partition_matrix size is creating an issue hence creating a partition_matrix_dict{}
                partition_matrix_dict = {}
                for node_index, supernode_index in whole_UGC_results.items():
                    # Append the node index to the list corresponding to the supernode index
                    if supernode_index not in partition_matrix_dict:
                        partition_matrix_dict[supernode_index] = [node_index]
                    else:
                        partition_matrix_dict[supernode_index].append(node_index)
                
                rr = 1 - len(partition_matrix_dict)/len(whole_UGC_results)
                print(f'Graph reduced by: {rr*100} percent.\nWe now have {len(partition_matrix_dict)} supernode, starting nodes were: {len(whole_UGC_results)}')

                super_nodes_labels = []
                super_node_features = []

                del(whole_UGC_results)
                # print(min(list(partition_matrix_dict.keys())),max(list(partition_matrix_dict.keys())))

                for key in list(partition_matrix_dict.keys()):
                    subnodes_in_supernode = partition_matrix_dict[key]
                    subnode_labels = concatenated_label[subnodes_in_supernode]
                    counts = np.bincount(subnode_labels)       
                    ## max ocuuring label, no_of_time it occurs
                    max_index, max_element_count = max(enumerate(counts), key=lambda x: x[1])

                    # print(subnode_labels,counts,max_index,max_element)
                    super_nodes_labels.append(max_index)
                    # print(np.sum(data[subnodes_in_supernode], axis=0)/len(subnodes_in_supernode))

                    ## check with only one feature just for testing
                    # super_node_features.append(cell_data[subnodes_in_supernode[0]])
                    
                    super_node_features.append(np.sum(cell_data[subnodes_in_supernode], axis=0)/len(subnodes_in_supernode))
    
                # super_nodes_labels, super_node_features, partition_matrix = get_partition_matrix_and_supernode_features(whole_UGC_results, cell_data, concatenated_label)
                
                super_node_features = np.array(super_node_features)
                print(super_node_features.shape)

                unique, counts = np.unique(super_nodes_labels, return_counts=True)
                # print(subnode_labels)
                print("super_nodes_labels ",unique, counts)

                # print(super_nodes_labels)

                whole_graph_end_time = time.time()
                print("partition time",whole_graph_end_time - whole_graph_start_time)
                information_dict['whole_graph_FACH'] = whole_graph_end_time - whole_graph_start_time
                
                print(" feature learning time",whole_graph_end_time - whole_graph_start_time)
                exit(1)
                
                file_name = 'results/whole_data_coarsened_' + data_name
                orignial_meld_pred, meld_accuracy_original_graph, original_data_phate, sample_likelihoods = run_meld(file_name,super_node_features, np.array(super_nodes_labels))
                
                # # print(super_nodes_labels)
                # plot_meld(file_name, num_classes, original_data_phate, sample_likelihoods.values, super_nodes_labels)
                # # exit(1)

                # #### sampled meld plot
                # number_of_subnodes = 15000
                # # print(super_nodes_labels)
                # sampled_labels_index, sampled_labels = sample_nodes_from_supernodes(partition_matrix_dict, number_of_subnodes, super_nodes_labels)
                # # print(sampled_labels_index)
                # # sampled_labels = np.array(super_nodes_labels)[np.array(sampled_labels_index)]
                
                # unique, counts = np.unique(sampled_labels, return_counts=True)
                # print("sampled_labels ",unique, counts)

                # sampled_subnode_features = cell_data[sampled_labels_index]
                # # print(sampled_labels)
                # _, _, sampled_data_phate, sampled_subnode_sample_likelihoods = run_meld(file_name,sampled_subnode_features, np.array(sampled_labels))
                
                # file_name = 'results/subnode_sampled_from_whole_data_coarsened_' + data_name
                # plot_meld(file_name, num_classes, sampled_data_phate, sampled_subnode_sample_likelihoods.values, sampled_labels)
                # exit(1)

                ##############

                # ### visulization of coarsened graph
                # percentage_to_use = 0.05  # Change this to adjust the percentage
                # num_rows_to_select = int(super_node_features.shape[0] * percentage_to_use)
                # selected_rows = np.random.choice(super_node_features.shape[0], num_rows_to_select, replace=False)

                
                # print("Sampling data for visulization purpose ",selected_rows.shape)
                # # Create the final matrix with selected rows
                # visulize_data_features = np.array(super_node_features)[selected_rows, :]
                # visulize_data_labels = np.array(super_nodes_labels)[selected_rows]

                # visulize_data(visulize_data_features, visulize_data_labels, coarsened_Data = True)
                #######
                
                # exit()
            ###################


            # print(cell_data)

            ##---------------------Meld with corrupted data labels
            # print("------------------------------------------------------------")
            # num_to_corrupt = int(0.8 * len(concatenated_label))

            # # Generate a list of random indices to corrupt
            # indices_to_corrupt = random.sample(range(len(concatenated_label)), num_to_corrupt)

            # # Replace the selected indices with random values (0 or 1)
            # for index in indices_to_corrupt:
            #     concatenated_label[index] = random.choice([0, 1])

            # orignial_meld_pred_with_corrupted_labels = run_meld(concatenated_data, concatenated_label)
            # print(" orignial_meld_pred_with_corrupted_labels ", orignial_meld_pred_with_corrupted_labels)

            #################################################
            
            feature_size = cell_data.shape[1]
            no_of_hash = 300
            hash_function = 'dot'
            projectors_distribution = 'normal'

            k_value = 5
            ## next 2 lines continue with the coarsened whole_dataset
            if args.coarsen_whole_dataset != False:
                after_coarsening_label_count = np.bincount(super_nodes_labels)
                print(after_coarsening_label_count)
                cell_data = np.concatenate((super_node_features, testing_data), axis=0)
                concatenated_label =  np.concatenate((super_nodes_labels, testing_label))

            after_coarsening_label_count_1 = np.bincount(concatenated_label)
            print(after_coarsening_label_count_1)

            whole_knn_time0 = time.time()            
            ## Learn Graph

            A_array = create_KNN_classification(cell_data, k_value, concatenated_label)
            np.fill_diagonal(A_array, 0)

            whole_knn_time1 = time.time()
            information_dict['whole_knn_time'] = whole_knn_time1 - whole_knn_time0

            print(data_name," dataset ratio ", dataset_ratio ," Knn graph learnt, size of our graph is ",A_array.shape," and time take to learn the knn graph ",whole_knn_time1-whole_knn_time0)
            # print(time4-start_time)

            ##### see the graph
            if args.plot_graph != False:
                subgraph = nx.from_numpy_array(A_array)
                # num_nodes_to_sample = 500
                # # Randomly select nodes from the original graph
                # sampled_nodes = random.sample(list(G.nodes()), num_nodes_to_sample)
                # # Create a subgraph by including only the sampled nodes and their adjacent edges
                # subgraph = G.subgraph(sampled_nodes)
                
                pos = nx.spring_layout(subgraph, seed=42)
                # print(concatenated_label[sampled_nodes])
                nx.draw(subgraph, with_labels=False, pos=pos, node_color=None, node_size=50)
                plt.show()
            
            edges_src = torch.from_numpy((np.nonzero(A_array))[0])
            edges_dst = torch.from_numpy((np.nonzero(A_array))[1])
            edge_index = torch.stack((edges_src, edges_dst))
            del A_array
            data = Data(x=torch.tensor(cell_data,dtype=torch.float32), edge_index = edge_index, y = torch.tensor(concatenated_label))
            
            #####
            ## Sota methods
            if args.sota_method != "None":
                sota_start_time = time.time()
                SCAL_utils.coarsening(data, args.sota_ratio, args.sota_method)
                sota_end_time = time.time()
                print("sota method ", args.sota_method, " dataset ", args.dataset, " C matrix finding time ",  sota_end_time-sota_start_time)
                exit(1)
            #####



            #
            # num_partitions = num_classes
            # epochs = 500
            # _, _ = diff_pool_clustering.gnn_clustering(data, epochs, num_partitions, dataset_name)
            #

            # #### checking the accuracy on the original/full dataset
            if args.train_on_original_data != False:# and large_datasets != True:
                hidden_units = 64
                learning_rate = 0.003
                decay = 0.0005
                epochs = args.epochs
                model_list = ['gcn']#,'graph_sage','gin','gat']
                full_gnn_value = {}
                # full_gnn_start = time.time()
                for model_type in model_list:
                    full_gnn_acc, full_gnn_val_acc_list = utils.train_on_original_dataset(data,num_classes,feature_size,hidden_units,learning_rate,decay,epochs, model_type, len(testing_nodes))
                    full_gnn_value[model_type] = [full_gnn_acc, full_gnn_val_acc_list]
                
                print("full_gnn_value ",full_gnn_value)
                information_dict['whole_Data_gnn_acc'] = full_gnn_value
                exit(1)
                # full_gnn_end = time.time()
                # full_gnn_time = full_gnn_end - full_gnn_start
            #continue
                # exit(1)
            #########################

            # test_split_percent = 0.2
            # data = utils.split(data,num_classes,test_split_percent) 

            ## new split function
            train_rate = 0.6
            val_rate = 0.2
            percls_trn = int(round(train_rate*len(data.y)/num_classes))
            val_lb = int(round(val_rate*len(data.y)))
            data = utils.random_splits(data, num_classes, percls_trn, val_lb)
            time2 = time.time()
            Bin_values = hashed_values(torch.tensor(cell_data), no_of_hash, feature_size,hash_function,projectors_distribution)  
            time3 = time.time()
            
            print("Done  with hased values function UGC time",time3-time2)
            if data_name == 'covid':
                list_bin_width = [70]#,160,175,190]#[25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110]#[3,3.5,4,4.3,4.7,5.2,5.7,6,6.5,7,7,5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14,5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20]
            elif data_name == 'preeclampsia':
                list_bin_width = [0.005]#[0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
            elif data_name == 'nkcell':
                list_bin_width = [0.01]#[0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
            elif dataset_name == 'xin': 
                list_bin_width = [1200]

            elif dataset_name == 'baron_mouse': 
                list_bin_width = [7]

            elif dataset_name == 'baron_human': 
                list_bin_width = [1]

            elif dataset_name == 'muraro': 
                list_bin_width = [6]

            elif dataset_name == 'segerstolpe': 
                list_bin_width = [200]

            elif dataset_name == 'amb': 
                list_bin_width = [20]

            elif dataset_name == 'tm': 
                list_bin_width = [0.1]

            elif dataset_name == 'zheng': 
                list_bin_width = [0.005]

            summary_dict = {}
            summary_dict = partition(list_bin_width,Bin_values,no_of_hash)
            time4 = time.time()

            print("Done  with partition function time should be roughly equal to ",(time4-time3)/len(list_bin_width))
            # exit(1) ## uncomment if you just want to note down the coarsening time of our method

            for bin_width in list_bin_width:
                time4 = time.time()
                current_bin_width_summary = summary_dict[bin_width]

                values = current_bin_width_summary.values()
                unique_values = set(values)
                rr = 1 - len(unique_values)/len(values)
                print(f'Graph reduced by: {rr*100} percent.\nWe now have {len(unique_values)} supernode, starting nodes were: {len(values)}')
                # exit(1)
                dict_blabla ={}
                C_diag = torch.zeros(len(unique_values), device= device)
                help_count = 0
                
                # i thinnk this can be improved
                # does this have a time complexity if O(N*v) ? i.e for each unique value searching each node hash value
                for v in unique_values:
                    C_diag[help_count],dict_blabla[help_count] = utils.get_key(v, current_bin_width_summary)
                    help_count += 1

                if args.gene_network == True:
                    handle_gene_network(original_gene_data, gene_list, C_diag, dict_blabla)
                    exit(1)                

                # P_hat is bool 2D array which represent nodes contained in supernodes 
                P_hat = torch.zeros((data.num_nodes, len(unique_values)), device= device)
                zero_list = torch.ones(len(unique_values), dtype=torch.bool)
                
                for x in dict_blabla:
                    if len(dict_blabla[x]) == 0:
                        print("zero element in this supernode",x)
                    for y in dict_blabla[x]:
                        P_hat[y,x] = 1
                        zero_list[x] = zero_list[x] and (not (data.train_mask)[y])
            
                same_label_nodes, super_nodes_labels = check_intra_node_similarity(P_hat.T.to('cpu'), np.array(concatenated_label))         
                # import pdb; pdb.set_trace()
                same_label_error = get_intra_node_error(P_hat.T.to('cpu'), super_nodes_labels,concatenated_label)
                information_dict['intra_node_accuracy'] = same_label_nodes/data.num_nodes
                information_dict['error'] = same_label_error/data.num_nodes
                
                file_name = 'results/Meld_coarsened_' + (str)(data_name)
                if args.meld_original != False:
                    meld_accuracy, coarsened_meld_time = get_meld_accuracy(P_hat.T.to('cpu'), orignial_meld_pred, cell_data, np.array(super_nodes_labels), file_name, original_data_phate, num_classes, sample_likelihoods.values, np.array(concatenated_label))
                    print("accuracy of UGC ",same_label_nodes/data.num_nodes, " error ", same_label_error/data.num_nodes, " meld_accuracy ", meld_accuracy)
                    information_dict['meld_accuracy_after_ugc_coarsened_graph'] = meld_accuracy
                    information_dict['coarsened_meld_time'] = coarsened_meld_time


                P_hat = P_hat.to_sparse()
                #dividing by number of elements in each supernode to get average value 
                P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))#.to(dtype=torch.float64)
                
                features =  data.x.to(device).to_sparse().to(dtype=torch.float32)

                # print(P.dtype)
                # print(features.to_dense().dtype)
                cor_feat = (torch.sparse.mm((torch.t(P)), features.to_dense()))#.to_sparse()
                i = data.edge_index.to(device)
                v = torch.ones(data.edge_index.shape[1]).to(device)
                shape = torch.Size([data.x.shape[0],data.x.shape[0]])
                g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device = device)
                g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))
                
                C_diag_matrix = np.diag(np.array(C_diag.to('cpu'), dtype = np.float32))#.to(device)
                #print("number of edges in the coarsened graph ",np.count_nonzero(g_coarse_adj.to_dense().to('cpu').numpy())/2)

                # next line only for GCN training 
                g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy() + C_diag_matrix - np.identity(C_diag_matrix.shape[0], dtype = np.float32)


                edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
                edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
                edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
                edge_index_corsen = torch.stack((edges_src, edges_dst)).to(device)
                edge_features = torch.from_numpy(edge_weight).to(device)

                print("edge_index_corsen ",edge_index_corsen.shape)
                Y = np.array(data.y.cpu())
                Y = utils.one_hot(Y,num_classes).to(device)
                Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)]).to(device)
                labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1).to(device)

                data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
                data_coarsen.edge_attr = edge_features

                final_time = time.time()
                print("time taken ", final_time - time2)
                ##### extra anaylsis
                # coarsening_quality_measure(C_diag, dict_blabla, Y, labels_coarse, num_classes)
                #####

                exit(1)

                print("checking unqiue classes in coarsened graphs")
                unique, counts = np.unique(labels_coarse.to('cpu'), return_counts=True)
                print(unique, counts)

                time5 = time.time()
                print('diff b/w t5 and t4 {}'.format(time5-time4))

                ##### getting cluster qualities
                # k_value = 10
                # A_array = create_KNN_classification(cor_feat.to('cpu'), k_value, labels_coarse.to('cpu'))
                # np.fill_diagonal(A_array, 0)

                # edges_src = torch.from_numpy((np.nonzero(A_array))[0])
                # edges_dst = torch.from_numpy((np.nonzero(A_array))[1])
                # edge_index = torch.stack((edges_src, edges_dst))
                # del A_array
                
                # data_coarsen.edge_index = edge_index
                # num_partitions = num_classes
                # epochs = 500
                # _, _ = diff_pool_clustering.gnn_clustering(data_coarsen, epochs, num_partitions, dataset_name)
                #######

                # exit()

                all_acc = []
                num_run = 5

                print("Done with UGC")

                time_taken_to_train_gcn = []
                for i in range(num_run):
                    global_best_val = 0
                    global_best_test = 0
                    best_val_acc = 0
                    best_epoch = 0

                    hidden_units = 16#args.hidden_units
                    learning_rate = 0.001#args.lr
                    decay = 0.0003#args.decay
                    epochs = 1000#args.epochs
                    

                    data_coarsen = data_coarsen.to(device)
                    edge_weight = torch.ones(data_coarsen.edge_index.size(1))
                    decay = decay

                    model_list = ['gcn']#,'graph_sage','gin','gat']
                    coarsened_gnn_value = {}
                    for model_type in model_list:
                        if model_type == 'graph_sage':
                            model = Graph_Sage.GraphSAGE(feature_size, hidden_units, num_classes)
                        elif model_type == 'gin':
                            model = GIN.GIN(feature_size, hidden_units, num_classes)
                        elif model_type == 'gat':
                            model = GAT.GAT(feature_size, hidden_units, num_classes)
                        else:
                            model = GCN.GCN_(feature_size, hidden_units, num_classes)

                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)

                        for epoch in range(epochs):
                            optimizer.zero_grad()
                            # print(data_coarsen.x)
                            # print(data_coarsen.edge_index)
                            # print(data_coarsen.edge_attr.float())
                            # print(data_coarsen.x.dtype)
                            # print(data_coarsen.edge_index.dtype)
                            # print(data_coarsen.edge_attr.float().dtype)
                            out = model(data_coarsen.x, data_coarsen.edge_index)#,data_coarsen.edge_attr.float()) 
                            pred = out.argmax(1)
                            criterion = torch.nn.NLLLoss()
                            # print(~zero_list)
                            # print(out)
                            # print(data_coarsen.y)
                            loss = criterion(out[~zero_list], data_coarsen.y[~zero_list]) 
                            optimizer.zero_grad() 
                            loss.backward()
                            optimizer.step()

                            # print(data.x.dtype)
                            # print(data.edge_index.dtype)
                            # print(data.edge_attr.dtype)

                            val_acc = val(model,data.to(device))
                            if best_val_acc < val_acc:
                                best_model = torch.save(model, 'best_model.pt') # model
                                best_val_acc = val_acc
                                best_epoch = epoch
                    
                            if epoch % 499 == 0:
                                # print('In epoch {}, loss: {:.3f}'.format(epoch, loss))
                                print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))

                        # model = model.to('cpu') #torch.load('best_model.pt')
                        model = torch.load('best_model.pt') #best_model
                        model.eval()
                        data = data#.to(device)
                        # pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
                        pred = model(data.x, data.edge_index).argmax(dim=1)
                        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                        
                        acc = int(correct) / int(data.test_mask.sum())
                        coarsened_gnn_value[model_type] = acc
                        print('--------------------------')
                        print('Accuracy on test data {:.3f}'.format(acc*100))

                        all_acc.append(acc)

                print(all_acc)
                average_Acc = np.sum(np.array(all_acc))/len(all_acc)
                std = np.std(np.array(all_acc))
                print("average_Acc :", average_Acc)
                print("std dev :",std)
                information_dict['coarsened_Data_gnn_acc'] = coarsened_gnn_value

                result_name = data_name + '_dataRatio_' + (str)(dataset_ratio) + '_coarseningRatio_' + (str)(rr)
                results[result_name] = information_dict

            result_name_gcn = data_name + '_dataRatio_' + (str)(dataset_ratio) + '_gcn_accuracy_full_dataset'
            # results[result_name_gcn] = full_dataset_gcn_acc

            file_path = 'results/new_results_for_cbm.json'

            # Write the data to the file in JSON format
            with open(file_path, 'r') as file:
                data = json.load(file)

            data.update(results)

            with open(file_path, 'w') as file:
                data = json.dump(data, file)
            
            final_time = time.time()

            print("total time taken is ",final_time - start_time)


    
if __name__ == "__main__":
    main()
