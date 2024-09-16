import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
import torch_geometric.transforms as T


# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()

        # Define the first GraphSAGE layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        # Define the second GraphSAGE layer
        self.conv4 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Apply the first GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)

        # Apply the second GraphSAGE layer
        x = self.conv4(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x

# Define the training loop
def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)
    # print(pred)
    # out = F.log_softmax(out, dim=1)
    # print(out)
    # pred = out.argmax(1)
    # print(data.train_mask.shape)
    # print(data.y.shape)
    # print("pred ",pred.shape)
    # print(pred[data.train_mask])
    # print(data.y[data.train_mask])
    # import pdb;pdb.set_trace()
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # Backward pass
    loss.backward()
    optimizer.step()

    return loss

# Define the evaluation function
def evaluate(model, data, train=False):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)

    pred = out.argmax(dim=1)
    if train==True:
        test_correct = pred[data.train_mask] == data.y[data.train_mask]
        test_acc = int(test_correct.sum()) / int(data.train_mask.sum())
    else:
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    return test_acc


def train_graphSage(num_features, num_classes, data):
    # dataset = Planetoid(root='/tmp/cora', name='Cora')
    
    model = GraphSAGE(num_features, 16, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=100e-4)

    for epoch in range(1, 1000):
        loss = train(model, data, optimizer)
        train_acc = evaluate(model, data, train=True)
        acc = evaluate(model, data)
        if epoch%50 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, train Acc: {train_acc:.4f}')

    return model
# dataset = Planetoid(root='data/Cora', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]


# train_graphSage(dataset.num_features, dataset.num_classes, data)
