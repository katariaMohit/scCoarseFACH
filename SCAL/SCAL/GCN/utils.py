from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import to_dense_adj
from graph_coarsening.coarsening_utils import *
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
import pandas as pd
from itertools import chain
import os
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import time

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
        if counter < 200:
            file_name = os.path.join(data_directory, file_name)
            print("file number ", counter, " file name ",file_name)

            if 'LPS+IFNa' not in file_name:
                counter += 1

                if data_name == 'covid':
                    file_data = pd.read_csv(file_name, delimiter="\t")
                else:
                    file_data = pd.read_csv(file_name)
                    if data_name == 'nkcell':
                        file_data = file_data.drop('label', axis=1)

                file_data = file_data.sample(frac=percentage_to_use, random_state=42)

                print(file_data.values.shape)
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

def create_KNN_classification(celldataset,k,celllabel):
    A_KNN=kneighbors_graph(celldataset, k,include_self=True, mode="distance")
    A_KNN= np.array(A_KNN.toarray())
    return A_KNN

def coarsening(dataset, coarsening_ratio, coarsening_method, coarsen_whole_dataset=False):
    data_path = {}
    data_repo_path = {}
    label_path = {}

    if dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=dataset)


    elif dataset == 'covid': 
        data_list = ['covid']
        data_repo_path['covid'] = r"..\..\..\data\Covid Dataset"
        data_path['covid'] = r"..\..\..\data\Covid Dataset\covid_dataset.csv"

    elif dataset == 'nkcell': 
        data_list = ['nkcell']
        data_repo_path['nkcell'] = r"..\..\..\data\NK Cell Dataset"
        data_path['nkcell'] = r"..\..\..\data\NK Cell Dataset\NKcell_dataset.csv"

    elif dataset == 'preeclampsia': 
        data_list = ['preeclampsia']
        data_repo_path['preeclampsia'] = r"..\..\..\data\Preeclampsia Dataset"
        data_path['preeclampsia'] = r"..\..\..\data\Preeclampsia Dataset\Han-FCS_file_list.csv"

    elif dataset == 'xin': 
        data_list = ['xin']
        data_repo_path['xin'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin"
        data_path['xin'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin\Filtered_Xin_HumanPancreas_data.csv"
        label_path['xin'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin\Labels.csv"

    elif dataset == 'baron_mouse': 
        data_list = ['baron_mouse']
        data_repo_path['baron_mouse'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse"
        data_path['baron_mouse'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse\Filtered_MousePancreas_data.csv"
        label_path['baron_mouse'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse\Labels.csv"

    elif dataset == 'baron_human': 
        data_list = ['baron_human']
        data_repo_path['baron_human'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human"
        data_path['baron_human'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human\Filtered_Baron_HumanPancreas_data.csv"
        label_path['baron_human'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human\Labels.csv"

    elif dataset == 'muraro': 
        data_list = ['muraro']
        data_repo_path['muraro'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro"
        data_path['muraro'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro\Filtered_Muraro_HumanPancreas_data.csv"
        label_path['muraro'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro\Labels.csv"

    elif dataset == 'segerstolpe': 
        data_list = ['segerstolpe']
        data_repo_path['segerstolpe'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe"
        data_path['segerstolpe'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe\Filtered_Segerstolpe_HumanPancreas_data.csv"
        label_path['segerstolpe'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe\Labels.csv"

    elif dataset == 'amb': 
        data_list = ['amb']
        data_repo_path['amb'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB"
        data_path['amb'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB\Filtered_mouse_allen_brain_data.csv"
        label_path['amb'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB\Labels.csv"

    elif dataset == 'tm': 
        data_list = ['tm']
        data_repo_path['tm'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM"
        data_path['tm'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM\Filtered_TM_data.csv"
        label_path['tm'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM\Labels.csv"

    elif dataset == 'zheng': 
        data_list = ['zheng']
        data_repo_path['zheng'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng"
        data_path['zheng'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng\Filtered_68K_PBMC_data.csv"
        label_path['zheng'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng\Labels.csv"

    if dataset not in ['covid','nkcell','preeclampsia'] or coarsen_whole_dataset == True:
        dataset_ratios = [1.0]
    else:
        # dataset_ratios = args.dataset_ratios
        if dataset == 'nkcell':
            dataset_ratios = [1.0]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]
        
        elif dataset == 'covid':
            dataset_ratios = [0.01]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]
        
        else:
            dataset_ratios = [0.01]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]    
        # dataset_ratios = [0.001, 0.002, 0.003, 0.004, 0.005]#, 0.01]
        # dataset_ratios = [1.0]#, 0.02,0.03, 0.04]

    for data_name in data_list:
        for dataset_ratio in dataset_ratios:           
            dir_path = data_repo_path[data_name]
            file_path = data_path[data_name]
            # print(dir_path)
            # print(file_path)     
            if dataset in ['covid','nkcell','preeclampsia']:
                concatenated_data, concatenated_label, num_classes = data_preprocessing(data_name, file_path, dir_path, dataset_ratio)
                unique, counts = np.unique(concatenated_label, return_counts=True)
                print(unique, counts)
                cell_data = concatenated_data.values
            
            # elif dataset in ['baron_5000', 'darmanis', 'deng', 'mECS', 'Kolod']:
            #     result = pyr.read_r(file_path) # also works for RData
            #     print(result)
            #     data=pd.DataFrame(result)
            #     print(data)
            #     exit(1)

            # elif dataset in ['PBMC']:
            #     with h5py.File(file_path, "r") as h5f:
            #         data = h5f["dataset"][:]
            #         print(data)
            #     exit(1)

            else:
                label_path = label_path[data_name]
                concatenated_data, concatenated_label, num_classes = data_preprocessing_scRNA_1(file_path, label_path, data_name)
                cell_data = concatenated_data.values
            
    k_value = 5
    A_array = create_KNN_classification(cell_data, k_value, concatenated_label)
    np.fill_diagonal(A_array, 0)

    edges_src = torch.from_numpy((np.nonzero(A_array))[0])
    edges_dst = torch.from_numpy((np.nonzero(A_array))[1])
    edge_index = torch.stack((edges_src, edges_dst))

    data = Data(x=cell_data, edge_index = edge_index, y = concatenated_label)
    
    start = time.time()
    # data = dataset[0]
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
    end = time.time()
    print('time taken for finding the components',end - start)
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


def load_data(dataset, candidate, C_list, Gc_list, exp, coarsen_whole_dataset=False):
    data_path = {}
    data_repo_path = {}
    label_path = {}

    if dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=dataset)
    
    elif dataset == 'covid': 
        data_list = ['covid']
        data_repo_path['covid'] = r"..\..\..\data\Covid Dataset"
        data_path['covid'] = r"..\..\..\data\Covid Dataset\covid_dataset.csv"

    elif dataset == 'nkcell': 
        data_list = ['nkcell']
        data_repo_path['nkcell'] = r"..\..\..\data\NK Cell Dataset"
        data_path['nkcell'] = r"..\..\..\data\NK Cell Dataset\NKcell_dataset.csv"

    elif dataset == 'preeclampsia': 
        data_list = ['preeclampsia']
        data_repo_path['preeclampsia'] = r"..\..\..\data\Preeclampsia Dataset"
        data_path['preeclampsia'] = r"..\..\..\data\Preeclampsia Dataset\Han-FCS_file_list.csv"

    elif dataset == 'xin': 
        data_list = ['xin']
        data_repo_path['xin'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin"
        data_path['xin'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin\Filtered_Xin_HumanPancreas_data.csv"
        label_path['xin'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Xin\Labels.csv"

    elif dataset == 'baron_mouse': 
        data_list = ['baron_mouse']
        data_repo_path['baron_mouse'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse"
        data_path['baron_mouse'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse\Filtered_MousePancreas_data.csv"
        label_path['baron_mouse'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Mouse\Labels.csv"

    elif dataset == 'baron_human': 
        data_list = ['baron_human']
        data_repo_path['baron_human'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human"
        data_path['baron_human'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human\Filtered_Baron_HumanPancreas_data.csv"
        label_path['baron_human'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Baron_Human\Labels.csv"

    elif dataset == 'muraro': 
        data_list = ['muraro']
        data_repo_path['muraro'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro"
        data_path['muraro'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro\Filtered_Muraro_HumanPancreas_data.csv"
        label_path['muraro'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Muraro\Labels.csv"

    elif dataset == 'segerstolpe': 
        data_list = ['segerstolpe']
        data_repo_path['segerstolpe'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe"
        data_path['segerstolpe'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe\Filtered_Segerstolpe_HumanPancreas_data.csv"
        label_path['segerstolpe'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Pancreatic_data\Segerstolpe\Labels.csv"

    elif dataset == 'amb': 
        data_list = ['amb']
        data_repo_path['amb'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB"
        data_path['amb'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB\Filtered_mouse_allen_brain_data.csv"
        label_path['amb'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\AMB\Labels.csv"

    elif dataset == 'tm': 
        data_list = ['tm']
        data_repo_path['tm'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM"
        data_path['tm'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM\Filtered_TM_data.csv"
        label_path['tm'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\TM\Labels.csv"

    elif dataset == 'zheng': 
        data_list = ['zheng']
        data_repo_path['zheng'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng"
        data_path['zheng'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng\Filtered_68K_PBMC_data.csv"
        label_path['zheng'] = r"..\..\..\scRNAseq_Benchmark_Datasets\scRNAseq_Benchmark_Datasets\Zheng\Labels.csv"

    if dataset not in ['covid','nkcell','preeclampsia'] or coarsen_whole_dataset == True:
        dataset_ratios = [1.0]
    else:
        # dataset_ratios = args.dataset_ratios
        if dataset == 'nkcell':
            dataset_ratios = [1.0]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]
        
        elif dataset == 'covid':
            dataset_ratios = [0.01]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]
        
        else:
            dataset_ratios = [0.01]#, 0.002, 0.003, 0.004, 0.005, 0.008,0.01]#[0.1,0.3, 0.5,1]    
        # dataset_ratios = [0.001, 0.002, 0.003, 0.004, 0.005]#, 0.01]
        # dataset_ratios = [1.0]#, 0.02,0.03, 0.04]

    for data_name in data_list:
        for dataset_ratio in dataset_ratios:           
            dir_path = data_repo_path[data_name]
            file_path = data_path[data_name]
            # print(dir_path)
            # print(file_path)     
            if dataset in ['covid','nkcell','preeclampsia']:
                concatenated_data, concatenated_label, num_classes = data_preprocessing(data_name, file_path, dir_path, dataset_ratio)
                unique, counts = np.unique(concatenated_label, return_counts=True)
                print(unique, counts)
                cell_data = concatenated_data.values
            
            # elif dataset in ['baron_5000', 'darmanis', 'deng', 'mECS', 'Kolod']:
            #     result = pyr.read_r(file_path) # also works for RData
            #     print(result)
            #     data=pd.DataFrame(result)
            #     print(data)
            #     exit(1)

            # elif dataset in ['PBMC']:
            #     with h5py.File(file_path, "r") as h5f:
            #         data = h5f["dataset"][:]
            #         print(data)
            #     exit(1)

            else:
                label_path = label_path[data_name]
                concatenated_data, concatenated_label, num_classes = data_preprocessing_scRNA_1(file_path, label_path, data_name)
                cell_data = concatenated_data.values
            
    k_value = 5
    A_array = create_KNN_classification(cell_data, k_value, concatenated_label)
    np.fill_diagonal(A_array, 0)

    edges_src = torch.from_numpy((np.nonzero(A_array))[0])
    edges_dst = torch.from_numpy((np.nonzero(A_array))[1])
    edge_index = torch.stack((edges_src, edges_dst))

    # print(torch.tensor(cell_data).dtype)
    # print(torch.tensor(edge_index).dtype)
    # exit(1)
    data = Data(x=torch.DoubleTensor(cell_data), edge_index = edge_index, y = torch.tensor(concatenated_label), train_mask=None, val_mask=None, test_mask=None)

##############  

    n_classes = len(set(np.array(data.y)))
    data = splits(data, n_classes, exp)
    train_mask = data.train_mask
    val_mask = data.val_mask
    labels = data.y
    features = data.x

    time0 = time.time()
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
        print("keep ", keep)
        print("C ", C_list[number])
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

            # print(" number of iterations ", C)

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

            coarsen_features = torch.cat([coarsen_features, torch.DoubleTensor(C.dot(H_features))], dim=0)
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

    coarsen_edge = torch.LongTensor(np.array([coarsen_row, coarsen_col]))
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()
    time1 = time.time()

    return data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, time1-time0


