# here we save the different GNN-classes for each dataset and also a function, which is called for training


# training function: num_layers, optimizer (loss,weight_decay), data, epochs =30,
# output: model
from torch.nn import Linear
from torch_geometric.nn import RGCNConv, to_hetero, FastRGCNConv
from datasets import create_hetero_ba_houses, PyGDataProcessor
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, to_hetero, RGCNConv
from torch_geometric.data import HeteroData
import os.path as osp


class HeteroGNNSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, nodetype_classify):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.nodetype_classify = nodetype_classify

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
        return self.lin(x_dict[self.nodetype_classify])


# Homogeneous RGCN

class RGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_relations, num_nodefeatures):
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels=7,  # TODO: Number of nodes of nodetype!!
                                  out_channels=16,
                                  num_relations=num_relations)
        self.conv2 = FastRGCNConv(16,
                                  out_channels=out_channels,
                                  num_relations=num_relations)
        # Adjusted dimensions
        # self.lin1 = Linear(hidden_channels, hidden_channels)
        # self.conv2 = RGCNConv(in_channels=hidden_channels,
        #                      out_channels=out_channels, num_relations=num_relations)
        # self.lin2 = Linear(hidden_channels, out_channels)
        print('RGCN model initialized.')

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)
        return x

# Heterogeneous RGCN


class HeteroRGCN(torch.nn.Module):
    def __init__(self, data, num_relations, num_nodefeatures, num_classes):
        super().__init__()
        # Ensure the input is a HeteroData object
        assert isinstance(data, HeteroData)

        # Define the homogeneous RGCN model
        model = RGCN(hidden_channels=64, out_channels=2,
                     num_nodefeatures=num_nodefeatures, num_relations=16)

        # Transform to heterogeneous model using data's metadata
        assert len(list(data.edge_index_dict.keys())
                   ) == 16, "some relations are missing"
        self.model = to_hetero(model, data.metadata(), debug=True)
        # self.lin = Linear(64, 2)

    def forward(self, x_dict, edge_index_dict, edge_type_dict) -> torch.Tensor:
        return self.model(x_dict, edge_index_dict, edge_type_dict)
        # return self.lin(x_dict[self.nodetype_classify])


class GNNDatasets():
    def __init__(self,
                 data,
                 num_layers=2,
                 type_to_classify=None,
                 optimizer=None,
                 model=None,
                 ):
        self.data = data
        self.num_layers = num_layers
        if type_to_classify is None:
            self.type_to_classify = self.data.type_to_classify
        else:
            self.type_to_classify = type_to_classify

        # ensure that the data has train, val and test splits
        if not hasattr(self.data[self.type_to_classify], 'train_mask'):
            dataprocessor = PyGDataProcessor(self.data, self.type_to_classify)
            self.data = dataprocessor.add_training_validation_test()
        if model is None:
            self.model = HeteroGNNSAGE(self.data.metadata(), hidden_channels=64, out_channels=2,
                                       nodetype_classify=self.type_to_classify, num_layers=self.num_layers)
        else:
            self.model = model
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        else:
            self.optimizer = optimizer

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        try:
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
        except TypeError:
            for node_type, features in self.data.x_dict.items():
                if features is None:
                    raise ValueError(
                        f"Node features for node type '{node_type}' are missing (None).")
            print('TypeError in train_epoch')
            relations_dict = {rel: i for i, rel in enumerate(
                self.data.edge_index_dict.keys())}
            # Correct the way edge_type_dict is created
            edge_type_dict = {rel: torch.ones(self.data.edge_index_dict[rel].size(
                1), dtype=torch.int64) for rel in relations_dict.keys()}
            self.data.edge_type_dict = edge_type_dict
            # Print edge_index_dict to see the available relations and their edge indices

            # Check for None values in edge_index_dict
            for relation, edge_index in self.data.edge_index_dict.items():
                if edge_index is None:
                    raise ValueError(
                        f"Edge indices for relation '{relation}' are missing (None).")

            out = self.model(
                self.data.x_dict, self.data.edge_index_dict, self.data.edge_type_dict)
        mask = self.data[self.type_to_classify].train_mask
        loss = F.cross_entropy(
            out[mask], self.data[self.type_to_classify].y[mask])
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        pred = self.model(self.data.x_dict,
                          self.data.edge_index_dict).argmax(dim=-1)
        accs = []
        for split in ['train_mask', 'val_mask', 'test_mask']:
            mask = self.data[self.type_to_classify][split]
            acc = (pred[mask] == self.data[self.type_to_classify].y[mask]
                   ).sum() / mask.size(dim=-1)
    # here mask.size not mask.sum(), because the mask is saved as the indices and not as boolean values
            accs.append(float(acc))
        return accs

    def train_model(self, epochs):
        self.model.train()
        for epoch in range(1, epochs):
            loss = self.train_epoch()
            train_acc, val_acc, test_acc = self.test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')


# ----------------- GNN for DBLP Dataset
# Paper: Heterogeneous Attention Network
# Code: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_conv_dblp.py
class HeteroGNN(torch.nn.Module):
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
    print('started training for ', modeldblp)
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.002, weight_decay=0.001)
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
