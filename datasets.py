import GPUtil
import torch_geometric
import torch
import os.path as osp


import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.data import HeteroData
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from sklearn.model_selection import train_test_split
import random
from collections import Counter
from torch_geometric.datasets import OGB_MAG, DBLP
import torch_geometric.transforms as T


# class Dataset:
"""
A class, which stores the principles data of a dataset in PyG.

Especially: Training, validation, test data. 

It transforms all input into one format (e.q. True/False as Training/Testinginstead of 0/1)

An object represents one (hdata) dataset.


Methods:
- __init__: initializes the dataset. This is an empty hdata object. 
- import_data: imports the data from a given path
- Tranform_from_dgl: transforms the data from dgl to PyG
- add_training_validation_test: adds the training, validation and test data to the dataset
- getter, setter for the hdata object.
"""


class HData:
    def __init__(self):
        # Initialize an empty hdata object
        self.data = HeteroData()
        self.type_to_classify = None

    def import_hdata(self, heterodata, type_to_classify=None):
        """
        Gets as input a heterodata object and checks, if train, validation and test data are included.
        And checks, if train, validation and test data are tensors with the indices of the nodes;
        not tensors of boolean true/False values. If so, it calls _convert_format to
        convert them to tensors with the indices of the nodes.
        """
        self.data = heterodata
        try:
            self.type_to_classify = heterodata.type_to_classify
        except Exception:
            self.type_to_classify = type_to_classify
        for split in ['train', 'val', 'test']:
            split_key = f'{split}_mask'
            if hasattr(heterodata[self.type_to_classify], split_key):
                self.data[self.type_to_classify][split_key] = getattr(heterodata[self.type_to_classify], split_key)
                self._convert_format()
            else:
                self.add_training_validation_test()

    def add_training_validation_test(self, training_percent=40, validation_percent=30, test_percent=30):

        # set the number of nodes of the to-be-classified type
        try:
            number_of_nodes = self.data[self.type_to_classify].num_nodes
        except Exception:
            number_of_nodes = self.data[self.type_to_classify].size()
            self.data[self.type_to_classify].num_nodes = number_of_nodes
        idx = torch.arange(number_of_nodes)

        if training_percent + validation_percent + test_percent != 100:
            if training_percent + validation_percent + test_percent == 1:
                training_percent = training_percent * 100
                validation_percent = validation_percent * 100
                test_percent = test_percent * 100
            elif training_percent < 0 or validation_percent < 0 or test_percent < 0:
                print("Error: Positive values were expected for training, validation and test sets")
                training_percent = 40
                validation_percent = 30
                test_percent = 30
            else:
                print("Error: It was expected to make a fair split into training, validation and test sets")
                training_percent = 40
                validation_percent = 30
                test_percent = 30

        train_idx, valid_and_test_idx = train_test_split(
            idx,
            train_size=0.01*training_percent,
        )
        valid_idx, test_idx = train_test_split(
            valid_and_test_idx,
            train_size=0.01*(validation_percent / (validation_percent+test_percent)),
        )
        self.data[self.type_to_classify].train_mask = torch.tensor(train_idx)
        self.data[self.type_to_classify].val_mask = torch.tensor(valid_idx)
        self.data[self.type_to_classify].test_mask = torch.tensor(test_idx)
        self._convert_format_train_val_test_mask()  # convert the format of the masks

    def _convert_format_train_val_test(self):
        # Helper method to convert data to required format (e.g., True/False to 1/0)
        # Implement the conversion logic here
        # First check, if the data is already in the right format
        # then check, if each node is in some training, validation or test set
        """
        This function converts the format of the training, validation and test data into tensors with the indices of the nodes.
        """
        total_nodes = self.data[self.type_to_classify].num_nodes
        for split in ['train', 'val', 'test']:
            split_key = f'{split}_mask'
            mask = getattr(self.data[self.type_to_classifnumber_of_nodes])
            if mask.dtype == torch.bool:
                new_mask = list()
                for ind in range(total_nodes):
                    if mask[ind]:
                        new_mask.append(ind)
                new_mask = torch.tensor(new_mask)
                self.data[self.type_to_classify][split_key] = new_mask
        set_train = set(self.data[self.type_to_classify].train_mask.tolist())
        set_val = set(self.data[self.type_to_classify].val_mask.tolist())
        set_test = set(self.data[self.type_to_classify].test_mask.tolist())
        intersection = set_train.intersection(set_val).intersection(set_test)
        if intersection:
            print("The training, validation and test data are not disjoint.")
            self.add_training_validation_test()

        # final test, if everything worked
        set_train = set(self.data[self.type_to_classify].train_mask.tolist())
        set_val = set(self.data[self.type_to_classify].val_mask.tolist())
        set_test = set(self.data[self.type_to_classify].test_mask.tolist())
        intersection = set_train.intersection(set_val).intersection(set_test)
        assert intersection == set(), "The training, validation and test data are not disjoint."

    def transform_from_dgl(self, dgl_data):
        # TODO: implement this method
        # Implement the logic to transform data from DGL to PyG format
        pass

    # Getter and setter methods
    @property
    def data(self):
        return self.data

    @data.setter
    def data(self, value):
        self.data = value
        self._convert_format_train_val_test()

    # Similar getters and setters for validation and test data can be added


# class BAGraphGraphMotifDataset():
"""
A class, which generates a Barabasi-Albert-Graph with a given number of nodes and edges.
Additionally, it can add motifs to the graph.


Object:


Methods:
- __init__: first calls the super class __init__ which initializes the dataset. This is an empty hdata object.
    The input is a dictionary which encodes a motif, or some pre-defined motif (like house). Optional input is the number of motifs, 
    which should be added; the number of nodes of the initial BA graph or an optional start graph.
    If no initial start graph is added, then a BA graph is created.
- add_motif: adds a motif to the graph. The input is a dictionary which encodes a motif, or some pre-defined motif 
    (like house).
- create_BAGraph: creates a Barabasi-Albert-Graph with a given number of nodes and edges. (using PyG)

"""


class BAGraphGraphMotifDataset():
    house_motif = {
        'labels': [1, 1, 2, 2, 3],
        'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2), (0, 3)],
    }

    def __init__(self, motif=None, num_motifs=0, num_nodes=None, start_graph=None):
        super().__init__()  # Assuming a superclass that initializes the dataset
        self.motif = motif
        self.num_motifs = num_motifs
        if start_graph is not None:
            self.orig_graph = start_graph
        else:
            if num_nodes is None:
                num_nodes = 400
            self.orig_graph = self.create_BAGraph(num_nodes)
        self.graph = self.orig_graph
        for _ in range(num_motifs):
            self.add_motif(motif)

    def add_motif(self, motif, graph=None):
        """
        Adds a motif to the graph (self.graph).
        Motifs are given by:
            - a list of nodes and their labels
            - a list of edges between nodes.

        First, a random node from the motif is chosen
        Second, a random node from the (possible already enhanced by motifs) BA Graph is chosen.
        """
        if isinstance(graph, nx) or isinstance(self.ba_graph, nx):
            pass
        else:
            raise Exception("The graph is not a networkx graph.")
        if graph is not None:
            num_graph_nodes = graph.number_of_nodes()
        elif hasattr(self, 'orig_graph'):
            num_graph_nodes = self.orig_graph.number_of_nodes()
            graph = self.orig_graph
        else:
            raise Exception("No graph was given and no BA graph was created yet or something else went wrong.")
        if motif == 'house':
            self.motif = BAGraphGraphMotifDataset.house_motif
            # select random node from bagraph and add an edge to the house motif
            start_node = random.randint(0, num_graph_nodes)
            end_node = random.randint(0, 4)+num_graph_nodes
            # Add nodes to graph
            for i, label in enumerate(self.motif['labels']):
                self.graph.add_node(i+num_graph_nodes, label=label)
            # Add edges
            for u_motif, v_motif in self.motif['edges']:
                u, v = u_motif + num_graph_nodes, v_motif + num_graph_nodes
                self.graph.add_edge(u, v)
            self.graph.add_edge(start_node, end_node)
            return self.graph
        else:
            raise Exception("This case is not implemented yet.")

    @staticmethod
    def create_BAGraph(num_nodes=400, num_new_edges=3):
        # Create a Barabasi-Albert graph using networkx
        # The number of edges to attach from a new node to existing nodes is 3
        ba_graph = nx.barabasi_albert_graph(num_nodes, num_new_edges)
        for node in ba_graph.nodes():
            ba_graph.nodes[node]['label'] = 0
        return from_networkx(ba_graph)  # Convert to PyTorch Geometric format

    # Additional methods and utility functions can be added as needed


# class HeteroBAMotifDataset(BAGraphMotifDataset, Dataset):
"""
Class, which makes a heterogenous graph out of a Barabasi-Albert-Graph with a given number of nodes and edges.
It takes a BAGraphMotifDataset as input and transfers each label into a node type.
Then, it assigns other nodes randomly to node types.

Input: 
- The previous label, which now should be classified.

Methods:
- __init__: first calls the super class __init__ which initializes the dataset. This is an empty hdata object.
    Additionally calls the super class __init__ of BAGraphMotifDataset.
    Then, it converts the labels into node types.
    Input: the node type, which should be classified.
- convert_labels_to_node_types: converts the labels into node types.
- add_random_types: adds random types to nodes in the graph, of the type which is also labelled. 
    All nodes in the motif get label 1, all nodes outside the motif get label 0.


"""

# class BAGraphCEMotifDataset(HeteroBAMotifDataset):
"""
Creates a Barabasi-Albert-Graph with a given number of nodes and edges.
Then it assigns the nodes random node types.
Then it creates some graphs from a CE
Additionally, it adds these graphs to the BA-graph

It uses fast-instance-checker to label the nodes afterwards.

Methods:
- __init__: Calls the constructor of the super class HeteroBAMotifDataset (output: non-empty hdata graph).
    also callable with a CE, which will be transformed to self.ce. If so, it also calls check_graph_ce.
- add_CE: adds a CE to the graph. The parameters are the Class Expression 
    and the number of nodes, which should be added to the graph.
    adds/ updates self.ce
- check_graph_ce: This labels all the nodes in the graph with satisfy the CE.



"""


def count_ints_total(input_list, intput):
    count = 0
    for element in input_list:
        if element == intput:
            count += 1
    return count


def count_ints_until_entry(input_list, intput, entry):  # works
    return count_ints_total(input_list[:entry], intput)


# utils: retrieve the second argument of the list_current_to_new_indices:
def new_index(list_of_pairs, index):
    for pair in list_of_pairs:
        if pair[0] == index:
            return pair[1]


def replace_random_zeros_with_one_or_three(input_list, prob_replace=0.07):
    output_list = []
    label_list = []
    counter = Counter(input_list)

    # Iterate over the counter and print the values and their frequencies
    for index_value in range(len(input_list)):
        if input_list[index_value] == 1:
            input_list[index_value] = 2
        elif input_list[index_value] == 2:
            input_list[index_value] = 1

    for value in input_list:
        if value == 0 and random.random() < 2*prob_replace:
            output_list.append(3)
            label_list.append(0)
        elif value == 0 and random.random() < prob_replace:
            output_list.append(1)
        elif value == 0 and random.random() < prob_replace:
            output_list.append(2)
        else:
            output_list.append(value)
            if value == 3:
                label_list.append(1)
    counter = Counter(input_list)
    # Iterate over the counter and print the values and their frequencies
    # for value, frequency in counter.items():
    #    print(f"Value: {value}, Frequency: {frequency}")
    return output_list, label_list


# it creates houses with labels 3-2-1 (top->bottom)
def create_hetero_ba_houses(not_house_nodes, houses):
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=not_house_nodes, num_edges=2),
        motif_generator='house',
        num_motifs=houses,
    )
    homgraph = dataset.get_graph()
    listnodetype = homgraph.y.tolist()
    listedgeindex = homgraph.edge_index.tolist()

    # randomly change some nodes of type 0 to type 3 or 1 and also retrieve a list of labels for nodes of type '3'
    listnodetype, label_list = replace_random_zeros_with_one_or_three(listnodetype, 0.1)

    number_of_each_type = []
    for i in range(4):
        number_of_each_type.append(count_ints_total(listnodetype, i))

    # [current index, new_index], where new_indes = count_ints_until_entry(... , label-of-current-index, current_index)
    list_current_to_new_indices = []
    for i in range(4):
        help_list_current_to_new_index = []
        for ind in range(len(listnodetype)):
            if listnodetype[ind] == i:
                help_list_current_to_new_index.append([ind, count_ints_until_entry(listnodetype, i, ind)])
        list_current_to_new_indices.append(help_list_current_to_new_index)

    hdata = HeteroData()
    # create nodes + feature 1
    list_different_node_types = [str(i) for i in list(set(listnodetype))]
    for nodetype in list_different_node_types:
        hdata[nodetype].x = torch.ones(number_of_each_type[int(nodetype)], 1)
    # asign labels to node 3:
    hdata['3'].y = torch.tensor(label_list)

    # create edges
    for type_start_index in range(len(list_different_node_types)):
        for type_end_index in range(type_start_index, len(list_different_node_types)):
            new_indices_start_list = []
            new_indices_end_list = []
            for start_node_index in range(len(listedgeindex[0])):
                # get nodetype of this label
                type_start = listnodetype[listedgeindex[0][start_node_index]]
                type_end = listnodetype[listedgeindex[1][start_node_index]]
                # check, if the labels are the wanted labels
                if type_start == type_start_index and type_end == type_end_index:
                    # get the new indizes:
                    look_up_list_start_node = list_current_to_new_indices[type_start]
                    look_up_list_end_node = list_current_to_new_indices[type_end]
                    new_start_index = new_index(look_up_list_start_node, listedgeindex[0][start_node_index])
                    new_end_index = new_index(look_up_list_end_node, listedgeindex[1][start_node_index])
                    new_indices_start_list.append(new_start_index)
                    new_indices_end_list.append(new_end_index)
            if new_indices_start_list and new_indices_end_list:
                # print(list_different_node_labels[label_start_index], [new_indices_start_list, new_indices_end_list])
                hdata[list_different_node_types[type_start_index], 'to', list_different_node_types[type_end_index]
                      ].edge_index = torch.tensor([new_indices_start_list, new_indices_end_list])
                if type_start_index != type_end_index:
                    hdata[list_different_node_types[type_end_index], 'to', list_different_node_types[type_start_index]
                          ].edge_index = torch.tensor([new_indices_end_list, new_indices_start_list])
    # only take nodes with labels
    idx = torch.arange(number_of_each_type[3])
    train_idx, valid_and_test_idx = train_test_split(
        idx,
        train_size=0.4,
    )
    valid_idx, test_idx = train_test_split(
        valid_and_test_idx,
        train_size=0.4,
    )
    hdata['3'].train_mask = torch.tensor(train_idx)
    hdata['3'].val_mask = torch.tensor(valid_idx)
    hdata['3'].test_mask = torch.tensor(test_idx)
    return hdata


# ------------------- DBLP Dataset


def initialize_dblp():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
    # We initialize conference node features with a single one-vector as feature:
    target_category_DBLP = 'conference'
    dataset = DBLP(path, transform=T.Constant(node_types=target_category_DBLP))
    target_category_DBLP = 'author'  # we want to predict classes of author
    # 4 different classes for author:
    #   database, data mining, machine learning, information retrieval.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = 'author'
    data = dataset[0]
    data = data.to(device)
    return data, target


# ------------------------ tests --------------------------------
print(GPUtil.getAvailable())
