import GPUtil
import torch_geometric
import torch
import os.path as osp
import copy
import itertools


import networkx as nx
import torch_geometric as pyg
import dgl
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from sklearn.model_selection import train_test_split
import random
from collections import Counter
from torch_geometric.datasets import DBLP
import torch_geometric.transforms as T


class PyGDataProcessor():
    """
    A class, which stores the principles data of a dataset in PyG.

    Especially: Training, validation, test data.

    It transforms all input into one format (e.q. True/False as Training/Testinginstead of 0/1)

    An object represents one (hdata) dataset.
    """

    def __init__(self, data=HeteroData(), type_to_classify=None):
        # Initialize an empty hdata object
        self._data = data
        self._type_to_classify = type_to_classify
        # if _data has a type_to_classify, then we can use it as a type_to_classify
        if hasattr(self._data, 'type_to_classify'):
            self._type_to_classify = self._data.type_to_classify

        else:
            self._data.type_to_classify = str(self._type_to_classify)
        if self._data.type_to_classify == 'None':
            self._type_to_classify = None

    def import_hdata(self, heterodata, type_to_classify=None):
        """
        Gets as input a heterodata object and checks, if train, validation and test data are included.
        And checks, if train, validation and test data are tensors with the indices of the nodes;
        not tensors of boolean true/False values. If so, it calls _convert_format to
        convert them to tensors with the indices of the nodes.
        """
        self._data = heterodata
        try:
            self._type_to_classify = heterodata.type_to_classify
        except Exception:
            self._type_to_classify = type_to_classify
        for split in ['train', 'val', 'test']:
            split_key = f'{split}_mask'
            if self._type_to_classify is not None:
                if hasattr(heterodata, self._type_to_classify):
                    if hasattr(heterodata[self._type_to_classify], split_key):
                        self._data[self._type_to_classify][split_key] = getattr(
                            heterodata[self._type_to_classify], split_key)
                        self._convert_format_train_val_test()
                    else:
                        self.add_training_validation_test()

    def add_training_validation_test(self, training_percent=40, validation_percent=30, test_percent=30):

        # set the number of nodes of the to-be-classified type
        self._type_to_classify = str(self._type_to_classify)
        try:
            number_of_nodes = self._data[self._type_to_classify].num_nodes
            if hasattr(self._data[self._type_to_classify], 'num_nodes'):
                number_of_nodes = self._data[self._type_to_classify].num_nodes
            else:
                number_of_nodes = self._data[self._type_to_classify].x.size()[0]
                self._data[self._type_to_classify].num_nodes = number_of_nodes
        except Exception:
            number_of_nodes = self._data[self._type_to_classify].size()
            self._data[self._type_to_classify].num_nodes = number_of_nodes
        assert isinstance(number_of_nodes, int), ("The number of nodes is not an integer.", number_of_nodes)
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
        try:
            train_idx, valid_and_test_idx = train_test_split(
                idx,
                train_size=0.01*training_percent,
            )
            valid_idx, test_idx = train_test_split(
                valid_and_test_idx,
                train_size=validation_percent / (validation_percent+test_percent),
            )
            self._data[self._type_to_classify].train_mask = torch.tensor(train_idx)
            self._data[self._type_to_classify].val_mask = torch.tensor(valid_idx)
            self._data[self._type_to_classify].test_mask = torch.tensor(test_idx)
            self._convert_format_train_val_test()  # convert the format of the masks
        except Exception:
            print("Not possible to split the data into training, validation and test sets, probably not enough data")

        return self._data

    def _convert_format_train_val_test(self):
        # Helper method to convert data to required format (e.g., True/False to 1/0)
        # Implement the conversion logic here
        # First check, if the data is already in the right format
        # then check, if each node is in some training, validation or test set
        """
        This function converts the format of the training, validation and test data into tensors with the indices of the nodes.
        """
        if hasattr(self._data[self._type_to_classify], 'num_nodes'):
            total_nodes = self._data[self._type_to_classify].num_nodes
        else:
            total_nodes = self._data[self._type_to_classify].x.size()[0]
            self._data[self._type_to_classify].num_nodes = total_nodes
        for split in ['train', 'val', 'test']:
            split_key = f'{split}_mask'
            mask = getattr(self._data[self._type_to_classify], split_key)
            if mask.dtype == torch.bool:
                new_mask = list()
                for ind in range(total_nodes):
                    if mask[ind]:
                        new_mask.append(ind)
                new_mask = torch.tensor(new_mask)
                self.data[self._type_to_classify][split_key] = new_mask
        set_train = set(self._data[self._type_to_classify].train_mask.tolist())
        set_val = set(self._data[self._type_to_classify].val_mask.tolist())
        set_test = set(self._data[self._type_to_classify].test_mask.tolist())
        intersection = set_train.intersection(set_val).intersection(set_test)
        if intersection:
            print("The training, validation and test data are not disjoint.")
            self.add_training_validation_test()

        # final test, if everything worked
        set_train = set(self._data[self._type_to_classify].train_mask.tolist())
        set_val = set(self._data[self._type_to_classify].val_mask.tolist())
        set_test = set(self._data[self._type_to_classify].test_mask.tolist())
        intersection = set_train.intersection(set_val).intersection(set_test)
        assert intersection == set(), "The training, validation and test data are not disjoint."

    # Getter and setter methods
    # Getter method for 'data'
    @property
    def data(self):
        return self._data

    # Setter method for 'data'
    @data.setter
    def data(self, value):
        self._data = value
        self._convert_format_train_val_test()

    @property
    def type_to_classify(self):
        """
        Getter for _type_to_classify.
        Returns the current value of _type_to_classify.
        """
        return self._type_to_classify

    @type_to_classify.setter
    def type_to_classify(self, value):
        """
        Setter for _type_to_classify.
        Sets the _type_to_classify to a new value.
        Additional checks or validations can be added here if required.
        """
        # Here you can add any validation or type checks if necessary
        self._type_to_classify = value

    # Similar getters and setters for validation and test data can be added


class GraphLibraryConverter():
    """
    This class has all functions to convert graphs from one library to another.
    Supported libraries: Networkx, PyG (heterogen and homogen), DGL

    The constructor saves the library (for DGL-Fromat) and a (heterogeneous) graph as an attribute.
    All methods are static.
    """

    def __init__(self, library, graph):
        self._library = library
        self._graph = graph  # in heterodata

    @staticmethod
    def networkx_to_homogen_pyg(graph):
        # Assuming node features are numeric and stored in a 'feature' attribute
        # If not, this part needs to be adjusted
        node_features = [graph.nodes[node]['feature'] for node in graph.nodes()]
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)

        # Extracting edge indices
        edge_list = list(graph.edges())
        edge_index = torch.tensor([[u, v] for u, v in edge_list], dtype=torch.long).t().contiguous()

        # Creating PyG Data object
        pyg_graph = Data(x=node_features_tensor, edge_index=edge_index)

        return pyg_graph

    @staticmethod
    def homogen_pyg_to_heterogen_pyg(graph):
        # Create a HeteroData object
        hetero_graph = HeteroData()

        # Assign a default node type and transfer node features
        # Assuming node features are in 'x'
        if 'x' in graph:
            hetero_graph['node_type'].x = graph.x

        # Assign a default edge type and transfer edge features and connections
        # Assuming edge_index is present
        if 'edge_index' in graph:
            hetero_graph['node_type', 'edge_type', 'node_type'].edge_index = graph.edge_index

            # Transfer edge features if present
            for key, value in graph.items():
                if key != 'x' and key != 'edge_index':
                    hetero_graph['node_type', 'edge_type', 'node_type'][key] = value

        return hetero_graph

    @staticmethod
    def networkx_to_heterogen_pyg(graph, edge_type='to'):
        # Implement conversion from Networkx graph to PyG graph

        # Scenario: Each label is a node type
        assert isinstance(graph, nx.Graph), "The graph is not a networkx graph."
        # Preprocessing: Ensure, all nodes have a label
        for node_id, attr in graph.nodes(data=True):
            if attr.get('label') is None:
                attr['label'] = '0'  # assuming default label is 0
            if not isinstance(attr['label'], str):
                attr['label'] = str(attr['label'])
        hetero_graph = HeteroData()
        labels = []
        for _, attr in graph.nodes(data=True):
            label = attr.get('label')  # Replace 'label' with your attribute key
            if label is not None:
                labels.append(label)
        labels = list(set(labels))
        labels = [str(x) for x in labels]
        labels = list(set(labels))

        # Step 1: count nodes of each label
        dict_nodecount = {}
        for label in labels:
            dict_nodecount[label] = 0
            for _, attr in graph.nodes(data=True):
                if attr.get('label') == label:
                    dict_nodecount[label] += 1
        # Step 2: create nodes

        for nodetype in labels:
            hetero_graph[str(nodetype)].x = torch.ones((dict_nodecount[nodetype], 1))

        # Step 3: create edges
        # 3.1: Create mapping old to new indices
        def count_nodes_with_label_until_id(G, label, node_id): return sum(1 for n, d in itertools.takewhile(
            lambda x: x[0] != node_id, G.nodes(data=True)) if d.get('label') == label)
        dict_current_to_new_indices = dict()
        # test, if all nodes have a label
        for node_id, attr in graph.nodes(data=True):
            assert attr.get('label') is not None, "Not all nodes have a label."
        for node_id, attr in graph.nodes(data=True):
            dict_current_to_new_indices[node_id] = count_nodes_with_label_until_id(graph, attr.get('label'), node_id)

        # 3.2: Create edges
        # iterate over all edges of graph: nx.Graph and transfer them to hetero_graph with the dict_current_to_new_indices: list
        for edge in graph.edges():
            u, v = edge
            label1 = str(graph.nodes[u].get('label'))  # Replace 'label' with your attribute key
            label2 = str(graph.nodes[v].get('label'))
            assert isinstance(u, int) and isinstance(v, int), "The nodes are not integers."
            u_new, v_new = dict_current_to_new_indices[u], dict_current_to_new_indices[v]
            # update hetero_graph:
            hetero_graph = GraphLibraryConverter.add_edge_to_hdata(
                hetero_graph, label1, edge_type, label2, u_new, v_new)

        # add number of nodes to hetero_graph
        for nodetype in labels:
            hetero_graph[str(nodetype)].num_nodes = dict_nodecount[nodetype]
        for nodetype in labels:
            assert isinstance(hetero_graph[str(nodetype)].num_nodes, int), "num_nodes did not produce an integer"
        # add number of num_node_types to hetero_graph
        hetero_graph.num_node_types = len(labels)

        # test if hetero_graph is bidirected:
        hetero_graph = GraphLibraryConverter.make_hdata_bidirected(hetero_graph)

        # ---- test, if everything worked
        # test, if all node types are strings
        for nodetype in hetero_graph.node_types:
            assert isinstance(nodetype, str), "The node types are not strings."
        # test, if it is not directed
        assert hetero_graph.is_undirected(), "The graph is not bidirected."

        # end tests

        return hetero_graph, dict_current_to_new_indices

    @staticmethod
    def pyg_to_networkx(graph):
        # Implement conversion from PyG graph to Networkx graph
        pass

    @staticmethod
    def networkx_to_dgl(graph):
        # Implement conversion from Networkx graph to DGL graph
        pass

    @staticmethod
    def dgl_to_networkx(graph):
        # Implement conversion from DGL graph to Networkx graph
        pass

    @staticmethod
    def dict_to_heterodata(dict):
        # dict must be of the form: {('node_type', 'edge_type', 'node_type'): (torch.tensor(), torch.tensor())}
        pass

    @staticmethod
    def dict_to_networkx(dict_orig):
        # graph_dict must be of the form: {('node_type', 'edge_type', 'node_type'): (torch.tensor(), torch.tensor())}
        # utils:
        graph_dict = copy.deepcopy(dict_orig)

        def remap_indices(new_indices_dict):
            remapped_dict = {}
            for node_type, index_dict in new_indices_dict.items():
                # Sort the values and remove duplicates
                unique_indices = sorted(set(index_dict.values()))
                # Create a mapping from old indices to a continuous range starting from 0
                continuous_mapping = {old_index: new_index for new_index, old_index in enumerate(unique_indices)}
                # Apply this mapping to the original indices
                remapped_dict[node_type] = {old_index: continuous_mapping[old_value]
                                            for old_index, old_value in index_dict.items()}
            return remapped_dict

        # steps:
        # 1. Get num_node_types as number of node types, and each type a number
        # 2. get max_num_nodes as maximal number of nodes of a type
        # 3. new indices: new_index = old_index + type_number*max_num_nodes
        # 4. make a dict: old_index -> new_index
        # 5. Change new_index: make them to the smallest natural number below the current index (st indices are 0,1,2,3,4,5,...)

        if not isinstance(graph_dict, dict):
            raise ValueError("graph_dict must be a dict")

        # Step 1: Get num_node_types and assign each type a number
        node_types = set()
        for edge_key in graph_dict.keys():
            node_types.update([edge_key[0], edge_key[2]])
        node_type_to_number = {node_type: i for i, node_type in enumerate(node_types)}
        num_node_types = len(node_types)

        # Step 2: Get max_num_nodes as maximal number of nodes of a type
        max_num_nodes = 0
        for edge_data in graph_dict.values():
            max_num_nodes = max(max_num_nodes, max(edge_data[0].max(), edge_data[1].max()).item() + 1)

        # Step 3: Calculate new indices
        new_indices_dict = {}
        for edge_key, edge_data in graph_dict.items():
            src_type, dst_type = edge_key[0], edge_key[2]
            src_type_number = node_type_to_number[src_type]
            dst_type_number = node_type_to_number[dst_type]
            if src_type not in new_indices_dict.keys():
                new_indices_dict[src_type] = {}
            if dst_type not in new_indices_dict.keys():
                new_indices_dict[dst_type] = {}

            for old_index in edge_data[0].tolist():
                old_index = int(old_index)
                new_index = old_index + src_type_number * max_num_nodes
                new_index = int(new_index)
                new_indices_dict[src_type][old_index] = new_index
            for old_index in edge_data[1].tolist():
                old_index = int(old_index)
                new_index = old_index + dst_type_number * max_num_nodes
                new_index = int(new_index)
                new_indices_dict[dst_type][old_index] = new_index

        new_indices_dict = remap_indices(new_indices_dict)
        # Step 4: Create a NetworkX graph
        G = nx.Graph()
        for edge_key, edge_data in graph_dict.items():
            src_type, dst_type = edge_key[0], edge_key[2]
            src_nodes, dst_nodes = edge_data
            for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
                new_src = new_indices_dict[src_type][src]
                new_dst = new_indices_dict[dst_type][dst]
                G.add_edge(new_src, new_dst)

        return G

    @staticmethod
    def add_edge_to_hdata(hetero_graph, start_type, edge_type, end_type, start_id: int, end_id: int):
        start_id_tensor = torch.tensor([start_id], dtype=torch.long)
        end_id_tensor = torch.tensor([end_id], dtype=torch.long)
        start_id, end_id = int(start_id), int(end_id)
        assert isinstance(hetero_graph, HeteroData), "The graph is not a heterogenous graph."

        if (start_type, edge_type, end_type) in hetero_graph.edge_types:
            list_ids_start, list_ids_end = [row.tolist()
                                            for row in hetero_graph[(start_type, edge_type, end_type)].edge_index]
            list_ids_start.append(start_id)
            list_ids_end.append(end_id)
            hetero_graph[(start_type, edge_type, end_type)].edge_index = torch.tensor([list_ids_start, list_ids_end])
            changes_made = True
        else:
            hetero_graph[start_type, edge_type, end_type].edge_index = torch.tensor([[start_id], [end_id]])
            changes_made = True

        return hetero_graph

    @staticmethod
    def make_hdata_bidirected(hetero_graph):
        """
        This makes a heterogenous graph bidirected and checks on validity: Each edge should exist in 2 directions.
        """
        for edge_type in hetero_graph.edge_types:
            start_type, relation_type, end_type = edge_type

            # Get the edge indices for this type
            edge_indices = hetero_graph[start_type, relation_type, end_type].edge_index
            start_indices, end_indices = edge_indices[0], edge_indices[1]

            # Iterate through the edges
            for start_id, end_id in zip(start_indices, end_indices):
                # Check if the reverse edge exists
                if (end_type, relation_type, start_type) in hetero_graph.edge_types:
                    reverse_edge_index = hetero_graph[end_type, relation_type, start_type].edge_index
                    if not any(end_id == reverse_edge_index[0][i] and start_id == reverse_edge_index[1][i] for i in range(len(reverse_edge_index[0]))):
                        # Add the reverse edge
                        # Assuming a function add_edge_to_hdata as defined previously
                        hetero_graph = GraphLibraryConverter.add_edge_to_hdata(
                            hetero_graph, end_type, relation_type, start_type, end_id.item(), start_id.item())
                else:
                    # Add the reverse edge
                    # Assuming a function add_edge_to_hdata as defined previously
                    hetero_graph = GraphLibraryConverter.add_edge_to_hdata(
                        hetero_graph, end_type, relation_type, start_type, end_id.item(), start_id.item())
        assert isinstance(hetero_graph, HeteroData), "The graph is not a heterogenous graph."
        assert hetero_graph.is_undirected(), f"The graph {hetero_graph} is not undirected."
        return hetero_graph


# class GenerateRandomHomogeneousGraph(): # Erbt von PyG-BAGraph, und anderen Graph Generators
"""
Erstellt über Parameter beliebige Random Graphs (BA, ...)
"""
# class GenerateRandomheterogeneousGraph(GenerateRandomHomogeneousGraph):
"""
Macht alles aus der Basisklasse GenerateRandomHomogeneousGraph für heterogene Graphen
"""


# class BAGraphGraphMotifDataset(): # Eventuell von GraphGeneration erben?: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/graph_generator/base.html#GraphGenerator
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
# class GenerateRandomGraph(): # Erbt von PyG-BAGraph, und anderen Graph Generators


class GenerateRandomGraph():  # getestet
    """
    Erstellt über Parameter beliebige Random Graphs (BA, ...) in beliebigen Formaten

    Methods:
    - __init__: generates an empty object (networkx)
    - create_BAGraph_nx: creates a Barabasi-Albert-Graph with a given number of nodes and edges. (using networkx)
    - create_BAGraph_pyg: creates a Barabasi-Albert-Graph with a given number of nodes and edges. (using PyG)
    """
    """
    Creates random graphs (BA, etc.) in various formats through parameters.
    """

    def __init__(self):
        """
        Generates an empty networkx object.
        """
        self.graph_nx = nx.Graph()

    @staticmethod
    def create_BAGraph_nx(num_nodes, num_edges):
        """
        Creates a Barabasi-Albert graph with a given number of nodes and edges using networkx.
        """
        graph_nx = nx.barabasi_albert_graph(num_nodes, num_edges)
        return graph_nx

    @staticmethod
    def create_BAGraph_pyg_homogen(num_nodes, num_edges):
        """
        Return homogeneous pyg graph
        Creates a Barabasi-Albert graph with a given number of nodes and edges using PyTorch Geometric (PyG).
        """
        graph_generator = BAGraph(num_nodes, num_edges)
        data = graph_generator()
        return data


class GraphMotifAugmenter():  # getestet
    """
    This class is designed to add motifs to a graph.
    The input is:
    - graph, 
    - a motif (given in homogeneous format), 
    - the number of times the motif should be added to the graph (num_motifs, default = 1)


    Methods:
    - __init__: initializes the class. Checks if the input graph is of the networkx format
        and if not converts it (using the GraphConverter class).

    There are some prededined motifs, like house, which can be added.

    Everything happens in networkx
    """
    house_motif = {
        'labels': [1, 1, 2, 2, 3],
        'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2), (0, 3)],
    }

    def __init__(self, motif='house', num_motifs=0, orig_graph=None):
        self.motif = motif
        self.num_motifs = num_motifs

        if orig_graph is not None:
            self.orig_graph = orig_graph
        else:
            num_nodes = 400
            num_edges = 3
            self.orig_graph = GenerateRandomGraph.create_BAGraph_nx(num_nodes, num_edges)
        self._graph = copy.deepcopy(self.orig_graph)
        self._list_node_in_motif_or_not = [0]*self.orig_graph.number_of_nodes()
        self._number_nodes_of_orig_graph = self.orig_graph.number_of_nodes()
        for _ in range(num_motifs):
            while True:
                try:
                    self._graph = self.add_motif(motif, self._graph)
                    break  # If the line executes without exceptions, exit the loop
                except Exception as e:
                    print(f"An exception occurred: {e}")
                    raise Exception("The graph is not connected or something with the motif is wrong.")

            # self._graph = self.add_motif(motif, self._graph)
            len_motif = 0
            try:
                len_motif = len(motif['labels'])
            except Exception:
                if motif == 'house':
                    len_motif = 5
            if motif == 'house':
                motif == GraphMotifAugmenter.house_motif
            self._list_node_in_motif_or_not.extend([1]*len_motif)

    @staticmethod
    def add_motif(motif, graph):  # getestet, aber geht trotzdem nicht
        """
        Adds a motif to the graph (self.graph).
        Motifs are given by:
            - a list of edges between nodes.

        First, a random node from the motif is chosen
        Second, a random node from the (possible already enhanced by motifs) BA Graph is chosen.
        """
        if isinstance(graph, nx.Graph):
            pass
        else:
            # TODO: convert to networkx
            raise Exception("The graph is not a networkx graph.")
        nodes_in_graph = len(graph.nodes)
        assert nodes_in_graph > 0, "The graph has no nodes."

        if motif == 'house':
            motif = GraphMotifAugmenter.house_motif
        if isinstance(motif, dict):
            # assetr tests, if the dictionary is correct
            nodes_in_motif = len(motif['labels'])
            assert 'labels' in motif, "The motif does not have labels."
            assert nodes_in_motif == len(motif['labels']), "The highest node in the motif is not the last node."

            # continue with the code
            # select random node from bagraph and add an edge to the house motif
            start_node = random.randint(0, nodes_in_graph-1)  # in ba_graph
            end_node = random.randint(0, nodes_in_motif-1)+nodes_in_graph  # in motif
            while end_node == start_node:
                end_node = random.randint(0, nodes_in_motif-1)+nodes_in_graph
            # Add nodes to graph
            assert 'labels' in motif, "The motif does not have labels."
            add_to = 0
            if 0 not in list(graph.nodes):
                add_to = 1
            current_num_nodes = len(graph.nodes)
            for i, label in enumerate(motif['labels']):
                graph.add_node(i+current_num_nodes+add_to, label=label)  # assuming, the nodes previously are 0,1,2,...
                assert current_num_nodes + i+1 == len(graph.nodes), ("The number of nodes did not increase by 1.", i,
                                                                     i+current_num_nodes+add_to, graph.nodes)
            for u_motif, v_motif in motif['edges']:
                u, v = u_motif + nodes_in_graph+add_to, v_motif + nodes_in_graph+add_to
                graph.add_edge(u, v)
            graph.add_edge(start_node, end_node)
            assert nx.is_connected(graph), "The graph is not connected."

            # Add edges
            # check if the labels worked
            labels = []
            for _, attr in graph.nodes(data=True):
                label = attr.get('label')
                labels.append(label)
            for label_motif in motif['labels']:
                assert label_motif in labels, "The label " + str(label_motif) + " is not in the graph."
            # end check

            # update the list, which nodes are in the motif and which are not
        else:
            raise Exception("This case is not implemented yet.")

        # Tests
        if motif == GraphMotifAugmenter.house_motif:
            labels = [str(node['label']) for _, node in graph.nodes(data=True) if 'label' in node]
            for i in range(1, 4):
                assert str(i) in labels, 'Label ' + str(i) + ' not in labels ' + str(labels)
        assert nx.is_connected(graph), "The graph is not connected."
        # End Tests
        return graph

    # getter and setter methods
    @property
    def number_nodes_of_orig_graph(self):
        """
        Getter for the number of nodes in the original graph.

        Returns:
        - int: The number of nodes in the original graph.
        """
        return self._number_nodes_of_orig_graph

    @property
    def graph(self):
        """
        Getter for the _graph attribute.

        Returns:
        - The graph object stored in the _graph attribute.
        """
        return self._graph


class HeteroBAMotifDataset():
    """
    Class, which makes a heterogenous graph out of a homogeneous graph with labels.
    It makes the labels to node types and adds random node types to the graph.
    All previous nodes without label get the lowest natural number (startint at 0) as a node type (called base-type), which is not used yet.




    Input: 
    - The previous label, which now should be the node type to be classified.
    - an instance of GraphMotifAugmenter

    Output:
    - a heterogenous graph in PyG format

    Methods:
    - __init__: 
        It converts the labels into node types.
        Input: the node type, which should be classified; the graph, which should be converted.
    - _convert_labels_to_node_types: converts the labels into node types.
    - _add_random_types: randomly changes node types of the base-type to other available types.
        Then it creates labels for each node of the type to be classified: 1 for nodes in a motif, 0 for nodes outside.
        Nodes outside the motif are all nodes with id less than number_nodes_of_orig_graph.
    """

    # TODO: This somehow does not work, see test_add_random_types
    def __init__(self, graph: nx.Graph,  type_to_classify=-1):
        self._augmenter = GraphMotifAugmenter()
        self._type_to_classify = type_to_classify
        self._graph = graph
        self._hdatagraph = HeteroData()
        self._edge_index = 'to'

        # resolve type_to_classify == -1:
        labels = []
        for _, attr in self._graph.nodes(data=True):
            label = attr.get('label')  # Replace 'label' with your attribute key
            if label is not None:
                labels.append(str(label))
        labels = list(set(labels))
        if self._type_to_classify == -1:
            self._type_to_classify = str(labels[-1])
        elif isinstance(self._type_to_classify, int):
            self._type_to_classify = str(self._type_to_classify)
        # set base label
        self._base_label = self._make_base_label(labels)

        # save labels
        labels.append(self._base_label)
        labels = list(set(labels))
        self.labels = labels
        self._hdatagraph.labels = labels

        # save type_to_classify into hdatagraph

        self._hdatagraph.type_to_classify = self._type_to_classify

    def _make_base_label(self, labels):
        """
        Makes a base label, which is not used yet.
        """
        # set base label
        if 0 not in labels and '0' not in labels:
            self._base_label = '0'
        else:
            labels_int = [int(label) for label in labels]
            self._base_label = str(max(labels_int)+1)
        for node in self._graph.nodes(data=True):
            node_id, attr = node
            if 'label' not in attr or attr['label'] is None or attr['label'] == 'None':
                self._graph.nodes[node_id]['label'] = self._base_label
        return self._base_label

    def _convert_labels_to_node_types(self, change_percent_labels=60):
        """
        Converts the labels into node types and adds this to self._hdatagraph.

        Steps:
        0. All nodes in the original graph get the node label 0 (or the lowest natural number, which is not used yet);
            this is the base-label. (in this function)
        1. Get the list of all node labels
            1.1. Get the list of all node types
            1.2. Randomly change nodes of the base-label to other available labels
            function: _add_random_types
        2. For each label, create a new node type
        3. Create a dictionary, st. for each node type: 
            3.1. For each node with this node type / label: get a new node-id, only based on this node-type
            3.2. Add the node-id to the dictionary
            Function: _convert_nxgraph_to_hdatagraph
        4. For each old edge, make the new edge with the new node-ids (between corresponding node types);
            use as edge_index: 'to'
            Add to self._hdatagraph
            Add Feature 1 to each node
            function: _convert_nxgraph_to_hdatagraph (together with 3.)
            This should now be finished and complete
        5. Add training, validation, test sets to self._hdatagraph for the node_type self._type_to_classify (default=-1)
            5.1.: Add labels to the nodes of the type to be classified: 1 for nodes in a motif, 0 for nodes outside.
            Function: _add_training_validation_test_sets (from class ... )
        """

        # Step 0
        # in constructor
        # Step 1
        # 1.1
        labels = self.labels
        if self._base_label in labels:
            labels.remove(self._base_label)
        # 1.2
        self._add_random_types(labels, self._base_label, change_percent=change_percent_labels)  # changes self._graph

        # Step 2-4
        labels.append(self._base_label)
        hdata_graph, dict_current_to_new_indices = GraphLibraryConverter.networkx_to_heterogen_pyg(
            self._graph, edge_type=self._edge_index)

        # Step 5.1:
        # retrieve a label_list for nodes inside / outside the motif graph
        if self._type_to_classify == -1:
            type_to_classify_str = str(labels[-2])  # -1 is self._base_label, which is not classified
        else:
            type_to_classify_str = str(self._type_to_classify)
        # tests
        assert type_to_classify_str != str(self._base_label), (type_to_classify_str, self._base_label, labels)
        node_types = hdata_graph.node_types
        node_types_str = [str(node) for node in node_types]
        assert type_to_classify_str in node_types_str, (type_to_classify_str, hdata_graph, self._graph.nodes(data=True))
        # assert False, ("This works: ", hdata_graph)
        # end tests
        label_list = [0]*hdata_graph[type_to_classify_str].num_nodes
        for node in self._graph.nodes(data=True):
            node_id, attr = node
            if str(attr['label']) == str(self._type_to_classify):
                if self._augmenter._list_node_in_motif_or_not[node_id] == 1:
                    new_node_id = dict_current_to_new_indices[node_id]
                    label_list[new_node_id] = 1

        hdata_graph[type_to_classify_str].y = torch.tensor(label_list)

        hetero_data_pygdataprocessor = PyGDataProcessor(hdata_graph, self._type_to_classify)
        hetero_data_pygdataprocessor.add_training_validation_test(
            training_percent=40, validation_percent=30, test_percent=30)

        # return the correct object
        return hetero_data_pygdataprocessor.data

    def _add_random_types(self, labels, base_label=None, change_percent=40):
        """
        Randomly changes node types of the base-type to other available types.
        Then it creates labels for each node of the type to be classified: 1 for nodes in a motif, 0 for nodes outside.
        Nodes outside the motif are all nodes with id less than number_nodes_of_orig_graph.

        Steps:
        1. Change labels of the base label to other labels by percentage change_percent; Stop if change_percent have been reached (iterate at random)
        """
        if base_label is None:
            base_label = str(self._base_label)
        assert base_label is not None, ("The base label is None.", base_label)

        # 1. Change labels of the base label to other labels
        number_labels = len(labels)
        nodes_with_data = list(self._graph.nodes(data=True))
        # first: Change all node without label to the base label
        for node_id, data in nodes_with_data:
            if data['label'] is None or data['label'] == 'None':
                data['label'] = base_label
            else:
                data['label'] = str(data['label'])

        new_labels = []
        for node_id, data in nodes_with_data:
            new_labels.append(data['label'])
        for i in range(0, len(labels)):
            assert labels[i] in new_labels, ("The label " + str(labels[i]) + " is not in the labels.", new_labels)
        # Second: Save all nodes without base_label in a list
        nodes_with_data = list(self._graph.nodes(data=True))
        node_ids_with_baselabel = []
        for node_id, data in nodes_with_data:
            if data['label'] == base_label:
                node_ids_with_baselabel.append(node_id)

        # random.shuffle(nodes_with_data)  # shuffle, st. the nodes with the base label are randomly distributed
        changes_nodes = 0
        num_nodes_total = self._augmenter.number_nodes_of_orig_graph
        # assert num_nodes_total < len(self._graph.nodes), self._graph.nodes
        for node_id, data in self._graph.nodes(data=True):
            if node_id in node_ids_with_baselabel:
                if random.random() < change_percent/100:
                    self._graph.add_node(node_id, label=labels[random.randint(0, len(labels)-1)])
                    # data['label'] = labels[random.randint(0, number_labels-1)]
                    changes_nodes += 1
            if changes_nodes >= (len(node_ids_with_baselabel)*change_percent)/100:
                break
        # tests

        # end tests
        return self._graph

    def print_statistics_to_dataset(dataset: HeteroData):
        """
        Adds statistics to the dataset.
        """
        # Number of nodes of each type, number of edges, number of motifs,
        # From node to classify: How many percent of the nodes are in a motif / have label 1?
        assert isinstance(dataset, HeteroData), "The dataset is not a heterogenous graph."
        # Number of nodes of each type
        print("Number of nodes of each type:")
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        elif hasattr(dataset, 'node_types'):
            labels = dataset.node_types
        else:
            raise Exception("NotImplemented:The dataset has no labels or node types.")
        for label in labels:
            print("Number of nodes of type " + str(label) + ": " +
                  str(dataset[label].num_nodes))
        print("Node type to be classified: " + str(dataset.type_to_classify))
        print("Number of nodes of each label")
        ground_truth_labels = dataset[dataset.type_to_classify].y.tolist()
        set_ground_truth_labels = set(ground_truth_labels)
        for value in list(set_ground_truth_labels):
            print("Number of nodes of label " + str(value) + ": " +
                  str(sum(1 for element in ground_truth_labels if element == value)))

    # getter and setter methods

    @property
    def augmenter(self):
        """
        Getter for the _augmenter attribute.

        Returns:
        - The GraphMotifAugmenter object stored in the _augmenter attribute.
        """
        return self._augmenter

    @augmenter.setter
    def augmenter(self, value):
        """
        Setter for the _augmenter attribute.

        Args:
        - value: The new GraphMotifAugmenter object to set.
        """
        # You can add any validation logic here if needed
        self._augmenter = value

    @property
    def type_to_classify(self):
        """
        Getter for the _type_to_classify attribute.

        Returns:
        - The type to classify.
        """
        return self._type_to_classify

    @type_to_classify.setter
    def type_to_classify(self, value):
        """
        Setter for the _type_to_classify attribute.

        Args:
        - value: The new type to classify.
        """
        # You can add any validation logic here if needed
        self._type_to_classify = value

    @property
    def edge_index(self):
        """
        Getter for the _edge_index attribute.

        Returns:
        - The edge_index.
        """
        return self._edge_index

    @edge_index.setter
    def edge_index(self, value):
        """
        Setter for the _edge_index attribute.

        Args:
        - value: The new edge_index.
        """
        # You can add any validation logic here if needed
        self._edge_index = value


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


# ------------------- Heterogeneous BAHouses Graphs ------------------------------


def count_ints_total(input_list, intput):
    count = 0
    for element in input_list:
        if element == intput:
            count += 1
    return count
    # short:
    # return sum(1 for element in input_list if element == input_int)


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
