# here we save the different GNN-classes for each dataset and also a function, which is called for training


# training function: num_layers, optimizer (loss,weight_decay), data, epochs =30,
# output: model
from torch_geometric.nn import HeteroConv, GCNConv
from datasets import create_hetero_ba_houses, initialize_dblp
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
import os.path as osp
from bashapes_model import HeteroGNNBA


# ----------------- GNN classes for basic training -----------------
class HeteroGNNTrainer:   # TODO: Test this class
    def __init__(self, model, data, learning_rate=0.01):
        self.model = model
        self.data = data
        self.data.type_to_classify = str(self.data.type_to_classify)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data, self.model = self.data.to(device), self.model.to(device)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(self.data.x_dict, self.data.edge_index_dict)

        # Assuming the task is node classification and labels are stored in data.y_dict
        # Modify the loss calculation based on your task
        # TODO: Remove tests
        assert isinstance(self.data, HeteroData)
        assert isinstance(self.data.type_to_classify, str)
        mask = self.data[self.data.type_to_classify].train_mask
        loss = F.cross_entropy(out[mask], self.data[self.data.type_to_classify].y[mask])
        loss.backward()
        self.optimizer.step()
        # return loss.item()   # TODO: Check if this works
        return float(loss)

    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
            pred = out.argmax(dim=-1)
            # Modify the evaluation based on your task
            correct = pred[mask] == self.data[self.data.type_to_classify].y[mask]
            accuracy = int(correct.sum()) / mask.size(dim=-1)
            return accuracy

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            loss = self.train_epoch()
            train_acc = self.evaluate(self.data[self.data.type_to_classify].train_mask)
            val_acc = self.evaluate(self.data[self.data.type_to_classify].val_mask)
            print(f"Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        return self.model

    def test(self):
        self.model.eval()
        test_acc = self.evaluate(self.data[self.data.type_to_classify].test_mask)
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc


class HeteroGNNModel(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, node_type, num_layers=2):
        super().__init__()
        self._nodetype = str(node_type)
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels, dropout=0.5)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = {key: F.leaky_relu(x)
                      for key, x in conv(x_dict, edge_index_dict).items()}
        return self.lin(x_dict[self._nodetype])

# Usage example
# metadata = data.metadata  # Metadata from HeteroData object
# model = HeteroGNNModel(metadata, hidden_channels=64, out_channels=num_classes, node_type='target_node_type')


# ----------------- GNN for DBLP Dataset
# Paper: Heterogeneous Attention Network
# Code: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_conv_dblp.py
class HeteroGNNDBLP(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = {key: F.leaky_relu(x)
                      for key, x in conv(x_dict, edge_index_dict).items()}
        return self.lin(x_dict['author'])


def train_epoch_dblp(modeldblp, datadblp, optimizer):
    modeldblp.train()
    optimizer.zero_grad()
    out = modeldblp(datadblp.x_dict, datadblp.edge_index_dict)
    mask = datadblp['author'].train_mask
    loss = F.cross_entropy(out[mask], datadblp['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


def test_dblp(modeldblp, datadblp):
    modeldblp.eval()
    pred = modeldblp(datadblp.x_dict, datadblp.edge_index_dict).argmax(dim=-1)
    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = datadblp['author'][split]
        acc = (pred[mask] == datadblp['author'].y[mask]).sum() / mask.sum()
        # acc = (pred[mask] == data['author'].y[mask]).sum() / mask.size(dim=-1)
        accs.append(float(acc))
    return accs


# TODO: put this in some function
# TODO: Rename into train_model
def train_model_dblp(modeldblp, datadblp, optimizer):
    modeldblp.train()
    for epoch in range(1, 200):
        loss = train_epoch_dblp(modeldblp, datadblp, optimizer)
        train_acc, val_acc, test_acc = test_dblp(modeldblp, datadblp)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')


def dblp_model(retrain):
    data = initialize_dblp()[0]
    model = HeteroGNN(data.metadata(), hidden_channels=32, out_channels=4,
                      num_layers=3)
    # model = to_hetero(model, data.metadata(), aggr='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datadblp, modeldblp = data.to(device), model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.001)
    path_name_saved = "content/models/"+'DBLP'
    is_file_there = osp.isfile(path_name_saved)
    if (is_file_there == True and retrain == False):
        print("using saved model")
        modeldblp.load_state_dict(torch.load(path_name_saved))
    else:
        print('training new model')
        train_model_dblp(modeldblp, datadblp, optimizer)
        print('new model is trained')
        PATH = "content/models/" + 'DBLP'
        print("File will be saved to: ", PATH)
        torch.save(model.state_dict(), PATH)
    modeldblp.eval()
    print('accuracy on DBLP: ', test_dblp(modeldblp, datadblp)[2])
    target = 'author'
    return modeldblp
