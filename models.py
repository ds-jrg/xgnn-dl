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
import copy


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
    def __init__(self, data, num_relations, num_bases, hidden_layers, out_channels):
        super().__init__()
        self.conv1 = FastRGCNConv(data.num_nodes, hidden_layers, num_relations,
                                  num_bases=num_bases)
        self.conv2 = FastRGCNConv(hidden_layers, out_channels, num_relations,
                                  num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        print('Debug edgeindex:', edge_index)
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


class RGCN_train():
    def __init__(self, data, type_to_explain) -> None:
        self.data = copy.deepcopy(data)
        num_relations = 16
        num_bases = 30
        hidden_layers = 32
        self.type_to_explain = type_to_explain
        if not hasattr(self.data, 'num_nodes'):
            sum_nodes = 0
            for nodetype in data.node_types:
                sum_nodes += data[nodetype].num_nodes
            self.data.num_nodes = sum_nodes
        if not hasattr(self.data, 'edge_index'):
            new_id = 0
            hetero_homo_dict = {}
            for nodetype in self.data.node_types:
                for id in range(data[nodetype].num_nodes):
                    hetero_homo_dict.update({f'{nodetype}_{id}': new_id})
                    new_id += 1
            # +1 was done in last step of for-loops above
            self.data.nodes = list(range(new_id))
            # create new edge_index

            edge_index = {}
            edge_type = []
            edge_count = 0
            train_idx = []
            train_y = []
            print('y values', self.data[type_to_explain].y)
            total_list_indices1, total_list_indices2 = [], []
            for edge, indices in self.data.edge_index_dict.items():
                list_indices = indices.tolist()
                for i in range(len(list_indices[0])):
                    list_indices[0][i] = hetero_homo_dict[f'{edge[0]}_{list_indices[0][i]}']
                    list_indices[1][i] = hetero_homo_dict[f'{edge[2]}_{list_indices[1][i]}']
                    edge_type.append(edge_count)

                total_list_indices1.extend(list_indices[0])
                total_list_indices2.extend(list_indices[1])
                edge_count += 1

            # add training, test
            train_idx = [hetero_homo_dict[f'{self.type_to_explain}_{i}']
                         for i in self.data[type_to_explain].train_mask]
            train_y = list(self.data[self.type_to_explain].y)
            train_y = [train_y[i]
                       for i in self.data[type_to_explain].train_mask]
            val_idx = [hetero_homo_dict[f'{self.type_to_explain}_{i}']
                       for i in self.data[type_to_explain].val_mask]
            val_y = list(self.data[self.type_to_explain].y)
            val_y = [val_y[i] for i in self.data[type_to_explain].val_mask]
            test_idx = [hetero_homo_dict[f'{self.type_to_explain}_{i}']
                        for i in self.data[type_to_explain].test_mask]
            test_y = list(self.data[self.type_to_explain].y)
            test_y = [test_y[i] for i in self.data[type_to_explain].test_mask]
            train_idx = torch.tensor(train_idx)
            test_idx = torch.tensor(test_idx)
            train_y = torch.tensor(train_y)
            test_y = torch.tensor(test_y)
            self.data.train_idx = train_idx
            self.data.val_idx = torch.tensor(val_idx)
            self.data.test_idx = test_idx
            self.data.train_y = train_y
            self.data.val_y = torch.tensor(val_y)
            self.data.test_y = test_y

            self.data.train_idx = torch.tensor(list(set(train_idx)))
            self.data.train_y = torch.tensor(
                [1 for i in range(len(train_idx))])
            self.data.edge_index = torch.tensor(
                [total_list_indices1, total_list_indices2])
            self.data.edge_type = torch.tensor(edge_type)

        self.model = RGCN(data=self.data,
                          num_relations=num_relations,
                          num_bases=num_bases,
                          hidden_layers=hidden_layers,
                          out_channels=2,
                          )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(None, self.data.edge_index, self.data.edge_type)
        loss = F.nll_loss(out[self.data.train_idx], self.data.train_y)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def train_model(self, epochs):
        self.model.train()
        for epoch in range(1, epochs):
            loss = self.train_epoch()
            train_acc, val_acc, test_acc = self.test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    @torch.no_grad()
    def test(self):
        self.model.eval()
        pred = self.model(None,
                          self.data.edge_index, self.data.edge_type).argmax(dim=-1)
        train_acc = (pred[self.data.train_idx] ==
                     self.data.train_y).float().mean()
        val_acc = (pred[self.data.val_idx] ==
                   self.data.val_y).float().mean()
        test_acc = (pred[self.data.test_idx] ==
                    self.data.test_y).float().mean()
        return train_acc, val_acc, test_acc


class HeteroRGCN(torch.nn.Module):
    """
    Not useable!!
    """

    def __init__(self, data, num_relations, num_nodefeatures, num_classes):
        super().__init__()
        # Ensure the input is a HeteroData object
        assert isinstance(data, HeteroData)

        sum_nodes = 0
        for nodetype in data.node_types:
            sum_nodes += data[nodetype].num_nodes

        # Define the homogeneous RGCN model
        self.model = RGCN(hidden_channels=64, out_channels=2,
                          num_nodes=sum_nodes, num_relations=16)

        # Transform to heterogeneous model using data's metadata
        assert len(list(data.edge_index_dict.keys())
                   ) == 16, "some relations are missing"
        # self.model = to_hetero(model, data.metadata(), debug=True)
        # self.lin = Linear(64, 2)

    def forward(self, x_dict, edge_index_dict, edge_type_dict) -> torch.Tensor:
        # print('debugging forward in HeteroRGCN', x_dict.keys(),
        #      edge_index_dict.keys(), edge_type_dict.keys())
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
                None, self.data.edge_index_dict, self.data.edge_type_dict)
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
