from dataclasses import dataclass
import dgl
import torch
import dgl.nn.pytorch as dglnn
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
import torch as th
from dgl.nn import SAGEConv
import os.path as osp
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, DBLP
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
import torch_geometric
from torch_geometric.nn import SAGEConv, to_hetero_with_bases, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.datasets import OGB_MAG
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph



hdata = HeteroData()

# Create two node types "paper" and "author" holding a feature matrix:
num_papers = 2
num_paper_features = 12
num_authors = 4
num_authors_features= 13
num_confs = 1
num_conf_features = 14
hdata['author'].x = torch.randn(num_authors, num_authors_features)
hdata['paper'].x = torch.randn(num_papers, num_paper_features)
hdata['conference'].x = torch.randn(num_confs, num_conf_features)
print(torch.randn(num_papers, num_paper_features))




# Create an edge type "(author, writes, paper)" and building the
# graph connectivity:
hdata['author', 'to', 'paper'].edge_index = torch.tensor([[2,3,3,1,0],[1,0,1,1,1]])
hdata['paper','to','conference'].edge_index = torch.tensor([[0,1],[0,0]])
hdata['paper', 'to', 'author'].edge_index = torch.tensor([[1,0,1,1,1],[2,3,3,1,0]])
hdata['conference','to','paper'].edge_index = torch.tensor([[0,0],[0,1]])
print('jetztiger Graph: ', hdata)

def visualize_heterodata(hd):
    g = torch_geometric.utils.to_networkx(hd.to_homogeneous())
    nx.draw(g, with_labels=True)
    plt.show()
#visualize_heterodata(hdata)









dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
data = dataset[0]
#print('OBG_MAG', data)

#create own dataset:
# Create two node types "paper" and "author" holding a feature matrix:

num_papers = 2
num_paper_features = 129
num_authors = 4
num_authors_features= 123
num_confs = 1
num_conf_features =7
hdata = HeteroData()
hdata['author'].x = torch.randn(num_authors, num_authors_features)
hdata['paper'].x = torch.randn(num_papers, num_paper_features)
hdata['conference'].x = torch.randn(num_confs, num_conf_features)

# Create an edge type "(author, writes, paper)" and building the
# graph connectivity:
hdata['author', 'to', 'paper'].edge_index = torch.tensor([[2,3,3,1,0],[1,0,1,1,1]])
hdata['paper', 'to', 'author'].edge_index = torch.tensor([[1,0,1,1,1],[2,3,3,1,0]])
hdata['paper','to','conference'].edge_index = torch.tensor([[0,1],[0,0]])
hdata['conference','to','paper'].edge_index = torch.tensor([[0,0],[0,1]])
#print('jetziger Graph: ', hdata)


#create a model on this data
#link: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html



class GNN(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.conv1 = HeteroConv({
                edge_type: SAGEConv((-1, -1), 32)
                for edge_type in metadata[1]
            })
        self.conv2 =HeteroConv({
                edge_type: SAGEConv((32, 32), 32)
                for edge_type in metadata[1]
            })
        self.lin = Linear(32, 2)
    def forward(self, x_dict, edge_index_dict):
        x_dict = {key: F.leaky_relu(x) for key, x in self.conv1(x_dict, edge_index_dict).items()}
        x_dict = {key: F.leaky_relu(x) for key, x in self.conv2(x_dict, edge_index_dict).items()}
        return  self.lin(x_dict['conference'])

model = GNN(hdata.metadata())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hdata, model = hdata.to(device), model.to(device)

#print(hdata.node_types)


homdata = hdata.to_homogeneous()



def count_ints_total(input_list, intput):
    count = 0
    for element in input_list:
        if element == intput:
            count += 1
    return count
def count_ints_until_entry(input_list, intput, entry):  #works
    return count_ints_total(input_list[:entry], intput)


#utils: retrieve the second argument of the list_current_to_new_indices:
def new_index(list_of_pairs, index):
    for pair in list_of_pairs:
        if pair[0] == index:
            return pair[1]


def create_hetero_ba_houses(not_house_nodes, houses):
    dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=not_house_nodes, num_edges=5),
    motif_generator='house',
    num_motifs=houses,
    )
    homgraph = dataset.get_graph()
    listnodelabel = homgraph.y.tolist()
    listedgeindex = homgraph.edge_index.tolist()

    number_of_each_label = []
    for i in range(4):
        number_of_each_label.append(count_ints_total(listnodelabel, i))

    
    #[current index, new_index], where new_indes = count_ints_until_entry(... , label-of-current-index, current_index)
    list_current_to_new_indices = []
    for i in range(4):
        help_list_current_to_new_index = []
        for ind in range(len(listnodelabel)):
            if listnodelabel[ind] == i:
                help_list_current_to_new_index.append([ind, count_ints_until_entry(listnodelabel, i, ind)])
        list_current_to_new_indices.append(help_list_current_to_new_index)


    hdata = HeteroData()
    #create nodes + feature 0
    list_different_node_labels = [str(i) for i in list(set(listnodelabel))]
    for label in list_different_node_labels:

        #hdata['author'].x = torch.randn(num_authors, num_authors_features)
        #hdata[label].x = [[0] for i in range(number_of_each_label[int(label)])]
        hdata[label].x = torch.zeros(number_of_each_label[int(label)], 1)


    #create edges
    for label_start_index in range(len(list_different_node_labels)):
        for label_end_index in range(label_start_index, len(list_different_node_labels)): 
            new_indices_start_list = []
            new_indices_end_list = []
            for start_node_index in range(len(listedgeindex[0])):
                # get nodetype of this label
                label_start = listnodelabel[listedgeindex[0][start_node_index]]
                label_end = listnodelabel[listedgeindex[1][start_node_index]]
                # check, if the labels are the wanted labels
                if label_start == label_start_index and label_end == label_end_index:
                    # get the new indizes:
                    look_up_list_start_node = list_current_to_new_indices[label_start]
                    look_up_list_end_node = list_current_to_new_indices[label_end]
                    new_start_index = new_index(look_up_list_start_node, listedgeindex[0][start_node_index])
                    new_end_index = new_index(look_up_list_end_node, listedgeindex[1][start_node_index])
                    new_indices_start_list.append(new_start_index)
                    new_indices_end_list.append(new_end_index)
            if new_indices_start_list and new_indices_end_list:
                #print(list_different_node_labels[label_start_index], [new_indices_start_list, new_indices_end_list])
                hdata[list_different_node_labels[label_start_index], 'to', list_different_node_labels[label_end_index]].edge_index = torch.tensor([new_indices_start_list, new_indices_end_list])
                if label_start_index != label_end_index:
                    hdata[list_different_node_labels[label_end_index], 'to', list_different_node_labels[label_start_index]].edge_index = torch.tensor([new_indices_end_list, new_indices_start_list])
    return hdata





bashapes = create_hetero_ba_houses(300,80)
print(bashapes)







        
        #for pair in list_current_to_new_indices[label_start_index]:
         #   print(pair)
        # add label_start_list, label_end_list to hdata

        



#visualize: check, if code actually produced houses









#now: Change this dataset into a dataset, where each node gets random types:

#list of types for top

#list of types for middle

#list of types for bottom

#creating a metagraph:
# creating one feature for each node


# creating node-edge-node-relations     # this is the hard part, as we have to retrieve the current graph structure and plug it in here 













#import sys
#sys.exit()