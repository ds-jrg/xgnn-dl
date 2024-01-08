import unittest
from datasets import PyGDataProcessor
from torch_geometric.datasets.graph_generator import BAGraph
import networkx as nx
import torch
from torch_geometric.data import HeteroData, Data
from datasets import GenerateRandomGraph, GraphMotifAugmenter, GraphLibraryConverter, HeteroBAMotifDataset
import torch_geometric
import copy
import random


class TestPyGDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = PyGDataProcessor()

    def test_initialization(self):
        self.assertIsInstance(self.processor.data, HeteroData)
        self.assertIsNone(self.processor._type_to_classify)

    def test_import_hdata_valid(self):
        # Create a dummy HeteroData object
        hdata = HeteroData()
        hdata['node_type'].x = torch.randn(3, 16)
        hdata['node_type'].train_mask = torch.tensor([True, False, True])
        hdata['node_type'].val_mask = torch.tensor([False, True, False])
        hdata['node_type'].test_mask = torch.tensor([False, False, True])
        hdata.type_to_classify = 'node_type'

        self.processor.import_hdata(hdata)
        self.assertEqual(self.processor.data, hdata)
        # Add more assertions to check if the masks are correctly imported and converted

    # Additional tests for other methods and edge cases
    def test_add_training_validation_test(self):
        # Create a dummy HeteroData object
        hdata = HeteroData()
        hdata['node_type'].x = torch.randn(3, 16)
        self.processor.import_hdata(hdata)
        self.processor._type_to_classify = 'node_type'

        # Call the method
        self.processor.add_training_validation_test(training_percent=40, validation_percent=30, test_percent=30)

        # Check if the masks are correctly set
        self.assertEqual(self.processor.data['node_type'].train_mask.size(0), 1)
        self.assertEqual(self.processor.data['node_type'].val_mask.size(0), 1)
        self.assertEqual(self.processor.data['node_type'].test_mask.size(0), 1)


class TestGraphLibraryConverter(unittest.TestCase):

    def setUp(self):
        # Setup code here (if needed)
        pass

    def test_networkx_to_heterogen_pyg_simple(self):
        # Create a sample NetworkX graph
        G = nx.Graph()
        G.add_node(1, label='A')
        G.add_node(2, label='B')
        G.add_edge(1, 2)

        # Convert to PyG heterogen graph
        hetero_graph, _ = GraphLibraryConverter.networkx_to_heterogen_pyg(G)

        # Assertions to check if conversion is correct
        self.assertIsInstance(hetero_graph, HeteroData)

        # Test, if all nodes ahve the correct type
        assert hetero_graph['A'].x.shape[0] == 1
        assert hetero_graph['B'].x.shape[0] == 1
        # Add more assertions as needed

    def test_networkx_to_heterogen_pyg_complex(self):
        # Create a sample NetworkX graph
        G = nx.Graph()
        G.add_node(1, label='A')
        G.add_node(2, label='B')
        G.add_node(3, label='A')
        G.add_node(4, label='B')
        G.add_edge(1, 2)
        G.add_edge(3, 4)
        G.add_edge(1, 3)
        G.add_edge(2, 4)

        # Convert to PyG heterogen graph
        hetero_graph, _ = GraphLibraryConverter.networkx_to_heterogen_pyg(G)

        # Assertions to check if conversion is correct
        self.assertIsInstance(hetero_graph, HeteroData)

        # Test, if all nodes ahve the correct type
        assert hetero_graph['A'].x.shape[0] == 2
        assert hetero_graph['B'].x.shape[0] == 2

        # make a more complex BA Graph + motif
        graph_nx = nx.barabasi_albert_graph(5, 3)
        augmenter = GraphMotifAugmenter('house', 1, graph_nx)
        graph_to_convert = augmenter.graph
        hetero_graph, _ = GraphLibraryConverter.networkx_to_heterogen_pyg(graph_to_convert)
        self.assertIsInstance(hetero_graph, HeteroData)

        # other assert tests
        self.assertEqual(hetero_graph['0'].x.shape[0], 5)
        self.assertEqual(hetero_graph['1'].x.shape[0], 2)
        self.assertEqual(hetero_graph['2'].x.shape[0], 2)
        self.assertEqual(hetero_graph['3'].x.shape[0], 1)
        self.assertEqual(hetero_graph.num_node_types, 4)


class TestGenerateRandomGraph(unittest.TestCase):

    def test_init(self):
        gen_graph = GenerateRandomGraph()
        self.assertIsInstance(gen_graph.graph_nx, nx.Graph)
        self.assertEqual(gen_graph.graph_nx.number_of_nodes(), 0)
        self.assertEqual(gen_graph.graph_nx.number_of_edges(), 0)

    def test_create_BAGraph_nx(self):
        num_nodes, num_edges = 10, 3
        gen_graph = GenerateRandomGraph()
        graph_nx = gen_graph.create_BAGraph_nx(num_nodes, num_edges)

        self.assertIsInstance(graph_nx, nx.Graph)
        self.assertEqual(graph_nx.number_of_nodes(), num_nodes)

        low_degree_nodes = [n for n in graph_nx.nodes() if graph_nx.degree(n) < num_edges]
        threshold = 0.3 * num_nodes
        error_message = f"Number of nodes with degree < {num_edges} exceeds 15% of total nodes. Count: {len(low_degree_nodes)}"
        self.assertTrue(len(low_degree_nodes) <= threshold, error_message)

    def test_create_BAGraph_pyg_homogen(self):
        num_nodes, num_edges = 10, 3
        gen_graph = GenerateRandomGraph()
        graph_pyg = gen_graph.create_BAGraph_pyg_homogen(num_nodes, num_edges)

        self.assertIsInstance(graph_pyg, Data, type(graph_pyg))
        self.assertEqual(graph_pyg.num_nodes, num_nodes)
        self.assertGreaterEqual(graph_pyg.num_edges, num_edges * (num_nodes-num_edges))


class TestGraphMotifAugmenter(unittest.TestCase):

    def setUp(self):
        # Set up a basic graph for testing
        self.basic_graph = nx.Graph()
        self.basic_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        self.basic_graph_save = copy.deepcopy(self.basic_graph)

    def test_initialization_with_graph(self):
        # Test initialization with a predefined graph
        augmenter = GraphMotifAugmenter('house', 1, self.basic_graph)
        self.assertEqual(augmenter.num_motifs, 1)
        self.assertIsNotNone(augmenter.orig_graph)
        self.assertEqual(augmenter._number_nodes_of_orig_graph, self.basic_graph_save.number_of_nodes())

    def test_initialization_without_graph(self):
        # Test initialization without providing a graph
        augmenter = GraphMotifAugmenter('house', 1)
        self.assertEqual(augmenter.num_motifs, 1)
        self.assertIsNotNone(augmenter.orig_graph)

    def test_add_motif(self):
        # Test adding a motif
        augmenter = GraphMotifAugmenter('house', 1, self.basic_graph)
        original_node_count = len(self.basic_graph.nodes)
        augmenter.add_motif(augmenter.house_motif, self.basic_graph)
        new_node_count = len(self.basic_graph.nodes)
        self.assertEqual(new_node_count, original_node_count + len(augmenter.house_motif['labels']))

    def test_invalid_graph_type(self):
        # Test initialization with an invalid graph type
        with self.assertRaises(Exception):
            invalid_graph = "not a graph"
            GraphMotifAugmenter('house', 1, invalid_graph)

    def test_unimplemented_motif(self):
        # Test adding an unimplemented motif
        augmenter = GraphMotifAugmenter('house', 1, self.basic_graph)
        with self.assertRaises(Exception):
            augmenter = GraphMotifAugmenter('unknown_motif', 1, self.basic_graph)
            augmenter.add_motif('unknown_motif', self.basic_graph)


class TestHeteroBAMotifDataset(unittest.TestCase):

    def setUp(self):
        # Setup a simple graph for testing
        self.simple_graph = nx.Graph()
        self.simple_graph.add_node(0, label='1')
        self.simple_graph.add_node(1, label='2')
        self.simple_graph.add_node(2)
        self.simple_graph.add_edge(0, 1)
        self.simple_graph.add_edge(1, 2)
        for i in range(3, 23):  # Start from 3 because 0, 1, and 2 are already added
            label = random.choice(['1', '2'])  # Randomly choose between label 1 and 2
            self.simple_graph.add_node(i, label=label)
            self.simple_graph.add_edge(i, random.choice(list(self.simple_graph.nodes)))
            self.simple_graph.add_edge(i, random.choice([0, 1]))

        # Setup an instance of GraphMotifAugmenter
        self.augmenter = GraphMotifAugmenter()  # Assuming this is a valid class

        # Setup an instance of HeteroBAMotifDataset
        testgraph = copy.deepcopy(self.simple_graph)
        self.dataset = HeteroBAMotifDataset(testgraph, 2)

        # setup a more complex graph
        complex_graph = GenerateRandomGraph.create_BAGraph_nx(num_nodes=50, num_edges=4)
        synthetic_graph_class = GraphMotifAugmenter(motif='house', num_motifs=50, orig_graph=complex_graph)
        synthetic_graph = synthetic_graph_class.graph
        self.complex_dataset = HeteroBAMotifDataset(synthetic_graph, -1)

    def test_conversion_complex_graph(self):

        # complex graph + added  motifs
        # check, if the traversion has still the same number of nodes
        # check, if there are nodes of all types

        # 1st: Test, if complex graph gets to a graph where all nodes are of type 0
        self.complex_dataset._convert_labels_to_node_types()

        pass

    def test_init(self):
        # Test the __init__ method
        self.assertIsInstance(self.dataset._graph, nx.Graph)
        self.assertIsInstance(self.dataset._hdatagraph, HeteroData)
        self.assertEqual(self.dataset._type_to_classify, '2')

    def test_convert_labels_to_node_types(self):
        # Test the _convert_labels_to_node_types method
        self.dataset._convert_labels_to_node_types()
        # Assertions to check if conversion is correct
        # Example: Check if base label is set correctly
        self.assertIn(self.dataset._base_label, ['0', '3'])  # Depending on your implementation

    def test_add_random_types(self):
        test_graph = copy.deepcopy(self.simple_graph)
        test_graph.nodes[2]['label'] = '0'
        # Test the _add_random_types method
        labels = ['1', '2']
        self.dataset._add_random_types(labels, '0', change_percent=40)
        # Assertions to check if random types are added correctly
        # Example: Check if labels are changed
        changed_labels = [data['label'] for _, data in self.dataset._graph.nodes(data=True)]
        self.assertIn('1', changed_labels)
        self.assertIn('2', changed_labels)

    def test_full_conversion_process(self):
        # Test the full conversion process from a NetworkX graph to a HeteroData object
        type_to_classify = 2
        test_graph = copy.deepcopy(self.simple_graph)
        dataset = HeteroBAMotifDataset(test_graph, type_to_classify)
        assert len(dataset.labels) == 3
        assert dataset._base_label == '0', (dataset._base_label, dataset.labels)
        self.assertEqual(dataset._type_to_classify, '2')
        self.assertEqual(len(dataset.labels), 3)
        self.assertEqual(dataset._base_label, '0')
        self.assertEqual(dataset._graph.number_of_nodes(), 23)
        hetero_data = dataset._convert_labels_to_node_types()
        # Assertions to check the final conversion
        # Example: Check if the output is a HeteroData object
        self.assertIsInstance(hetero_data, HeteroData)

        # test, if the correct number of features are created


if __name__ == '__main__':
    unittest.main()


# ------------------------------ playground ------------------------------
def define_hetero_data():
    hetero_data = HeteroData()

    # Define node features for two types of nodes
    hetero_data['type1'].x = torch.randn(3, 16)  # 3 nodes of type1 with 16 features each
    hetero_data['type2'].x = torch.randn(2, 16)  # 2 nodes of type2 with 16 features each

    # Define edge connections
    # Connecting nodes of 'type1' with nodes of 'type2'
    edge_index = torch.tensor([[0, 1, 2, 0, 1], [0, 0, 1, 1, 1]], dtype=torch.long)
    hetero_data['type1', 'type2'].edge_index = edge_index
    # self.hetero_data = hetero_data
    return hetero_data


print(define_hetero_data())


graph_generator = BAGraph(num_nodes=300, num_edges=5)
print(graph_generator)
graph = nx.Graph()
for i, label in enumerate([1, 2, 3, 4, 5]):
    graph.add_node(i, label=label)
for node in graph.nodes(data=True):
    node_id, attr = node
    label = attr['label']
    print(f"Node ID: {node_id}, Label: {label}")
print(graph.number_of_nodes())
