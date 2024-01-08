import unittest
from models import HeteroGNNTrainer, HeteroGNNModel
import torch
from torch_geometric.data import HeteroData
from datasets import HeteroBAMotifDataset, GenerateRandomGraph, GraphMotifAugmenter


class TestHeteroGNNTrainer(unittest.TestCase):
    """
    Tests HeteroGNNTrainer class on datasets created by datasets.py
    """

    def setUp(self):
        # Create a dummy model and data
        type_to_classify = '3'
        self.type_to_classify = type_to_classify

        def test_new_datasets(num_nodes, num_motifs):
            # test the new datasets
            # create BA Graph
            ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(num_nodes=num_nodes, num_edges=2)
            motif = 'house'

            synthetic_graph_class = GraphMotifAugmenter(motif=motif, num_motifs=num_motifs, orig_graph=ba_graph_nx)
            synthetic_graph = synthetic_graph_class.graph
            dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
            dataset = dataset_class._convert_labels_to_node_types()
            print('Data the GNN is trained on: ', dataset)
            return dataset
        self.test_new_datasets = test_new_datasets
        self.data = test_new_datasets(500, 50)
        self.model = HeteroGNNModel(self.data.metadata(), hidden_channels=16, out_channels=2,
                                    node_type=self.data.type_to_classify, num_layers=2)
        self.trainer = HeteroGNNTrainer(self.model, self.data)

    def test_initialization(self):
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.data, self.data)
        self.assertEqual(self.trainer.learning_rate, 0.01)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Adam)
        self.assertIsInstance(self.trainer.data, HeteroData)

    def test_train_epoch(self):
        loss = self.trainer.train_epoch()
        self.assertIsInstance(loss, float)

    def test_evaluate(self):
        """
        Define a mask from the correct size and see if it works
        """
        pass

    def test_train(self):
        pass

    def test_test(self):
        self.trainer.test()

    def test_complete_model(self):
        """
        Test the complete model
        """
        self.trainer.train(epochs=15)
        self.trainer.test()

        # dummy dataset
        out_of_sample_data = self.test_new_datasets(5, 1)
        out = self.trainer.model(out_of_sample_data.x_dict, out_of_sample_data.edge_index_dict)
        # assert False, out
        assert out.shape[1] == 2
        assert out.shape[0] == out_of_sample_data.x_dict[self.type_to_classify].shape[0]
        assert self.trainer.test() > 0.3, "Model did not learn at all"


# ---------------- run testing
if __name__ == '__main__':
    unittest.main()
