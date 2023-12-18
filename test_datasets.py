import unittest
from datasets import HData
from torch_geometric.datasets.graph_generator import BAGraph
import networkx as nx


class TestHData(unittest.TestCase):
    def setUp(self):
        self.hdata = HData()

    def test_import_hdata(self):
        # Test case 1: Importing heterodata with train, validation, and test data
        heterodata = ...
        self.hdata.import_hdata(heterodata)
        self.assertEqual(self.hdata.data, heterodata)
        self.assertEqual(self.hdata.type_to_classify, heterodata.type_to_classify)

        # Test case 2: Importing heterodata without train, validation, and test data
        heterodata = ...
        type_to_classify = ...
        self.hdata.import_hdata(heterodata, type_to_classify)
        self.assertEqual(self.hdata.data, heterodata)
        self.assertEqual(self.hdata.type_to_classify, type_to_classify)

    def test_add_training_validation_test(self):
        # Test case 1: Adding training, validation, and test data with valid percentages
        self.hdata.add_training_validation_test(training_percent=40, validation_percent=30, test_percent=30)
        self.assertIsNotNone(self.hdata.data[self.hdata.type_to_classify].train_mask)
        self.assertIsNotNone(self.hdata.data[self.hdata.type_to_classify].val_mask)
        self.assertIsNotNone(self.hdata.data[self.hdata.type_to_classify].test_mask)

        # Test case 2: Adding training, validation, and test data with invalid percentages
        self.hdata.add_training_validation_test(training_percent=50, validation_percent=30, test_percent=20)
        self.assertIsNotNone(self.hdata.data[self.hdata.type_to_classify].train_mask)
        self.assertIsNotNone(self.hdata.data[self.hdata.type_to_classify].val_mask)
        self.assertIsNotNone(self.hdata.data[self.hdata.type_to_classify].test_mask)

    def test_convert_format_train_val_test(self):
        # Test case 1: Converting format of training, validation, and test data
        self.hdata._convert_format_train_val_test()
        self.assertEqual(self.hdata.data[self.hdata.type_to_classify].train_mask.dtype, torch.tensor)
        self.assertEqual(self.hdata.data[self.hdata.type_to_classify].val_mask.dtype, torch.tensor)
        self.assertEqual(self.hdata.data[self.hdata.type_to_classify].test_mask.dtype, torch.tensor)

        # Test case 2: Converting format of training, validation, and test data with non-disjoint sets
        self.hdata.data[self.hdata.type_to_classify].train_mask = torch.tensor([0, 1, 2])
        self.hdata.data[self.hdata.type_to_classify].val_mask = torch.tensor([2, 3, 4])
        self.hdata.data[self.hdata.type_to_classify].test_mask = torch.tensor([4, 5, 6])
        self.hdata._convert_format_train_val_test()
        self.assertNotEqual(self.hdata.data[self.hdata.type_to_classify].train_mask.tolist(), [0, 1, 2])
        self.assertNotEqual(self.hdata.data[self.hdata.type_to_classify].val_mask.tolist(), [2, 3, 4])
        self.assertNotEqual(self.hdata.data[self.hdata.type_to_classify].test_mask.tolist(), [4, 5, 6])

    def test_transform_from_dgl(self):
        # Test case: Transforming data from DGL to PyG format
        dgl_data = ...
        self.hdata.transform_from_dgl(dgl_data)
        self.assertEqual(self.hdata.data, expected_pyg_data)

    def test_data_getter_setter(self):
        # Test case: Getter and setter for data property
        data = ...
        self.hdata.data = data
        self.assertEqual(self.hdata.data, data)


# if __name__ == '__main__':
#    unittest.main()

# ------------------------------ playground ------------------------------

graph_generator = BAGraph(num_nodes=300, num_edges=5)
print(graph_generator)
graph = nx.Graph()
for i, label in enumerate([1,2,3,4,5]):
    graph.add_node(i, label=label)
for node in graph.nodes(data=True):
    node_id, attr = node
    label = attr['label']
    print(f"Node ID: {node_id}, Label: {label}")
print(graph.number_of_nodes())
