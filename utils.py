import torch
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
import GCN
import GAT, GIN, Graph_Sage


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)#, device=index.device)
    mask[index] = 1
    return mask

def get_key(val, g_coarsened):
  KEYS = []
  for key, value in g_coarsened.items():
    if val == value:
      KEYS.append(key)
  return len(KEYS),KEYS


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

def random_splits_large_data(data, num_classes, percls_trn, val_lb, num_testing_nodes, seed=42):
    index=[i for i in range(0,data.y.shape[0]-num_testing_nodes)]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
            
    val_idx = [i for i in index if i not in train_idx]
    test_idx = [i for i in range(data.y.shape[0]-num_testing_nodes, data.y.shape[0])]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

def split(data, num_classes, split_percent):
    indices = []
    num_test = (int)(data.x.shape[0] * split_percent / num_classes)
    for i in range(num_classes):
        index = (data.y == i).nonzero().reshape(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    
    test_index = torch.cat([i[:num_test] for i in indices], dim=0)
    val_index = torch.cat([i[num_test:int(num_test*1.5)] for i in indices], dim=0)
    train_index = torch.cat([i[int(num_test*1.5):] for i in indices], dim=0)
    
    # print("num_classes ",num_classes)
    # print(test_index)
    # print(val_index)

    data.train_mask = index_to_mask(train_index, size=data.x.shape[0])
    data.val_mask = index_to_mask(val_index, size=data.x.shape[0])
    data.test_mask = index_to_mask(test_index, size=data.x.shape[0])

    # train_count = data.train_mask.sum().item()
    # test_count = data.test_mask.sum().item()
    # val_count = data.val_mask.sum().item()

    # print("Number of entries in the training set:", train_count)
    # print("Number of entries in the test set:", test_count)
    # print("Number of entries in the validation set:", val_count)

    return data

def val(model,data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc

def train_on_original_dataset(data, num_classes, feature_size, hidden_units, learning_rate, decay, epochs, model_type, num_testing_nodes=None):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if model_type == 'graph_sage':
    model = Graph_Sage.GraphSAGE(feature_size, hidden_units, num_classes)
  elif model_type == 'gin':
    model = GIN.GIN(feature_size, hidden_units, num_classes)
  elif model_type == 'gat':
    model = GAT.GAT(feature_size, hidden_units, num_classes)
  else:
    model = GCN.GCN_(feature_size, hidden_units, num_classes)

  model = model.to(device)
  data = data.to(device)
  

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)
  
  # test_split_percent = 0.2
  # data = split(data,num_classes,test_split_percent)

  ## new split check if anything changes 

  if num_testing_nodes == 0:
    train_rate = 0.6
    val_rate = 0.2

    percls_trn = int(round(train_rate*len(data.y)/num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    
    data = random_splits(data, num_classes, percls_trn, val_lb)
  else:
    train_rate = 0.8
    val_rate = 0.2

    percls_trn = int(round(train_rate*(len(data.y)-num_testing_nodes)/num_classes))
    val_lb = int(round(val_rate*(len(data.y)-num_testing_nodes)))

    data = random_splits_large_data(data, num_classes, percls_trn, val_lb, num_testing_nodes) 

  
  if data.edge_attr == None:
    edge_weight = torch.ones(data.edge_index.size(1))
    data.edge_attr = edge_weight

  val_acc_list = []
  for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)#,data.edge_attr.float())
    pred = out.argmax(1)
    criterion = torch.nn.NLLLoss()
    # print(pred)
    # # print(out)
    # print(data.y)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].long()) 
    optimizer.zero_grad()
    # print(loss)
    loss.backward()
    optimizer.step()
    best_val_acc = 0
    
    val_acc = val(model,data)
    if best_val_acc < val_acc:
        best_model = torch.save(model, 'full_best_model.pt') #model
        best_val_acc = val_acc
  
    if epoch % 499 == 0:
        print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))
    
    if epoch % 20 == 0:
        val_acc_list.append(val_acc)

  model = torch.load('full_best_model.pt') #best_model 
  model.eval()
  data = data#.to(device)
  pred = model(data.x, data.edge_index).argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc = int(correct) / int(data.test_mask.sum())
  
  print('--------------------------')
  print('Accuracy on test data {:.3f}'.format(acc*100))

  return acc, val_acc_list
