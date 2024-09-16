import torch
import diff_pool_gnn_model
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
import cluster_metric

def tsne_plot_clustering_compare(labels_1 , labels_2, X ):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Define a fixed color map
    color_palette = sns.color_palette("viridis", 10)
    fixed_color_map = {0: color_palette[0],
                      1: color_palette[1],
                      2: color_palette[2],
                      3: color_palette[3],
                      4: color_palette[4],
                      5: color_palette[5],
                      6: color_palette[6],
                      7: color_palette[7],
                      8: color_palette[8],
                      9: color_palette[9]}

    # Plot t-SNE with first set of labels
    plt.figure(figsize=(12, 5))
    print(labels_1.shape)
    print(len(labels_2))

    plt.subplot(1, 2, 1)
    for label in np.unique(labels_1):
        plt.scatter(X_tsne[labels_1 == label, 0], X_tsne[labels_1 == label, 1], 
                    color=fixed_color_map[label], label=f'Cluster {label}')
    plt.title('Clustering Method 1')
    plt.legend()

    # Plot t-SNE with second set of labels
    plt.subplot(1, 2, 2)
    for label in np.unique(labels_2):
        plt.scatter(X_tsne[labels_2 == label, 0], X_tsne[labels_2 == label, 1], 
                    color=fixed_color_map[label], label=f'Cluster {label}')
    plt.title('Clustering Method 2')
    plt.legend()

    plt.show()

def t_sne_visualize_graph(data_subset,targets,name): 
  tsne = TSNE(n_components=2,n_iter=1000,perplexity=(int)(data_subset.shape[0]/2))
  tsne_results = tsne.fit_transform(data_subset)
  
  fig, ax = plt.subplots(figsize=(7, 7))
  ax.scatter(
      tsne_results[:, 0],
      tsne_results[:, 1],
      c=targets,
      s=15,
      cmap="jet",
      alpha=0.7,
  )
  ax.set(
      aspect="equal",
      xlabel="$X_1$",
      ylabel="$X_2$",
      )
  plt.savefig(name)
  plt.show()


def gnn_clustering(data, epochs, num_clusters, dataset_name):
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)
    
    if device.type == 'cuda':
        torch.cuda.manual_seed(1)

    data = data.to(device)

    num_features = data.x.shape[1]

    # model = diff_pool_gnn_model.Net([32]*2, "ReLU", num_features, num_clusters, [16], "ReLU").to(device)
    model = diff_pool_gnn_model.Sage([32]*2, "ReLU", num_features, num_clusters, [16], "ReLU").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        _, loss = model(data.x, data.edge_index)#, data.edge_weight)
        loss.backward()
        optimizer.step()
        return loss.item()


    @torch.no_grad()
    def test():
        model.eval()
        clust, _ = model(data.x, data.edge_index)#, data.edge_weight)
        g_adj = to_dense_adj(data.edge_index)[0]
        if dataset_name == 'word2vec':
            pairwise_accuracy = 0
            nmi = 0
        else:
            pairwise_accuracy = cluster_metric.pairwise_accuracy(data.y.cpu(), clust.max(1)[1].cpu())
            nmi = NMI(clust.max(1)[1].cpu(), data.y.cpu())

        conductance = cluster_metric.conductance(g_adj.to(device), clust.max(1)[1].to(device))
        modularity = cluster_metric.modularity(g_adj.to(device), clust.max(1)[1].to(device))
        #return NMI(clust.max(1)[1].to(device), data.y.to(device)), conductance, pairwise_accuracy, modularity
        return nmi, conductance, pairwise_accuracy, modularity
    

    for epoch in range(1, epochs):
        train_loss = train()
        nmi, conductance, pairwise_accuracy, modularity = test()
        if epoch%100 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, ' f'NMI: {nmi:.3f}, '  f'Conductance: {conductance:.3f}, '  f'pairwise_accuracy: {pairwise_accuracy:.3f}, '  f'modularity: {modularity:.3f}')

    clust, _ = model(data.x, data.edge_index)#, data.edge_weight)
    # new_G_figname_full = f"results_for_visulization/Clustering_full_start_graph_" + dataset_name + ".pdf"
    # t_sne_visualize_graph(np.array(data.x.cpu()), data.y.cpu(), new_G_figname_full)
    # new_G_figname = f"results_for_visulization/Clustering_" + dataset_name + ".pdf"
    # t_sne_visualize_graph(np.array(data.x.cpu()), clust.max(1)[1].cpu(), new_G_figname)

    tsne_plot_clustering_compare(data.y.cpu() , clust.max(1)[1].cpu(), data.x.cpu() )

    # print(clust.shape)
    cluster_values = clust.max(1)[1].tolist()

    # t_sne_visualize_graph(data, cluster_values, new_G_figname)

    # print(np.unique(cluster_values))
    
    def find_partitions(arr):
        index_dict = {}
        for i, num in enumerate(arr):
            if num not in index_dict:
                # index_dict[num] = [str(i)]
                index_dict[num] = [i]
            else:
                # index_dict[num].append(str(i))
                index_dict[num].append(i)
        return index_dict

    partitions = find_partitions(cluster_values)

    return list(partitions.values()), model