import pandas as pd
import os
from itertools import chain
import json
import time
import numpy as np

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
                # print(file_data)
                concatenated_data = pd.concat([concatenated_data, file_data], axis=0,ignore_index=True)

    concatenated_label =  list(chain.from_iterable(concatenated_label)) 
    concatenated_label = np.array(concatenated_label)

    # print(concatenated_data.head())  # You can display the concatenated data or further process it
    # print(concatenated_data.values.shape)
    # print("label information ")
    # print(concatenated_label)

    return concatenated_data, concatenated_label, num_classes

temp_time = time.time()

# data_list = ['preeclampsia']#,'nkcell']#,'covid']
data_list = ['nkcell']


data_path = {}
data_repo_path = {}

data_repo_path['covid'] = "data\Covid Dataset"
data_path['covid'] = "data\Covid Dataset\covid_dataset.csv"

data_repo_path['nkcell'] = r"data\NK Cell Dataset"
data_path['nkcell'] = r"data\NK Cell Dataset\NKcell_dataset.csv"

data_repo_path['preeclampsia'] = r"data\Preeclampsia Dataset"
data_path['preeclampsia'] = r"data\Preeclampsia Dataset\Han-FCS_file_list.csv"


dataset_ratios = [0.001]#, 0.002, 0.003, 0.004, 0.005]#, 0.01]
# dataset_ratios = [0.01, 0.02,0.03, 0.04]
results = {}
print(data_list)
print(dataset_ratios)
for data_name in data_list:

    for dataset_ratio in dataset_ratios:

        dir_path = data_repo_path[data_name]
        file_path = data_path[data_name]
        # print(dir_path)
        # print(file_path)
        print()
        concatenated_data, concatenated_label, num_classes = data_preprocessing(data_name, file_path, dir_path, dataset_ratio)

        cell_data = concatenated_data.values
        cell_label = concatenated_label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import phate
import scprep
import meld
import sklearn
import tempfile
import os
import scanpy as sc

# making sure plots & clusters are reproducible
np.random.seed(42)

phate_op = phate.PHATE(n_jobs=-1)
data_phate = phate_op.fit_transform(concatenated_data)

sample_cmap = {
    0: '#fb6a4a',
    1: '#08519c'
}
scprep.plot.scatter2d(data_phate, c=concatenated_data['label'], cmap=sample_cmap, 
                      legend_anchor=(1,1), figsize=(6,5), s=10, label_prefix='PHATE', ticks=False)

# beta is the amount of smoothing to do for density estimation
# knn is the number of neighbors used to set the kernel bandwidth
meld_op = meld.MELD(beta=67, knn=7)
sample_densities = meld_op.fit_transform(concatenated_data, sample_labels=concatenated_data['label'])

fig, axes = plt.subplots(1,2, figsize=(11,6))

for i, ax in enumerate(axes.flatten()):
    density = sample_densities.iloc[:,i]
    scprep.plot.scatter2d(data_phate, c=density,
                          title=density.name,
                          vmin=0, 
                          ticks=False, ax=ax)
    
fig.tight_layout()

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

sample_likelihoods = normalize_densities(sample_densities)#, np.array(concatenated_data['label']))
print(sample_likelihoods)

from sklearn.metrics import accuracy_score

y_pred = [1 if prob[1] > prob[0] else 0 for prob in sample_likelihoods.values]

# Calculate accuracy
accuracy = accuracy_score(concatenated_data['label'], y_pred)
print(accuracy)
