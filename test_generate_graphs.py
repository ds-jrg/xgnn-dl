import unittest
from unittest.mock import MagicMock
from generate_graphs import get_gnn_outs
from datasets import HeteroBAMotifDataset, GenerateRandomGraph, GraphMotifAugmenter
from models import HeteroGNNModel, HeteroGNNTrainer


class TestGetGnnOuts(unittest.TestCase):

    def setUp(self):
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
        out_of_sample_data = test_new_datasets(5, 1)
        data = test_new_datasets(510, 50)
        self.hd = out_of_sample_data
        # model
        model = HeteroGNNModel(data.metadata(), hidden_channels=16, out_channels=2,
                               node_type=type_to_classify, num_layers=2)
        self.trainer = HeteroGNNTrainer(model, data)
        self.trainer.train(epochs=20)
        self.trainer.test()
        self.model = self.trainer.model

    def test_pipeline(self):
        out = self.model(self.hd.x_dict, self.hd.edge_index_dict)
        result = get_gnn_outs(self.hd, self.model, -1)
        assert isinstance(result, float)


if __name__ == '__main__':
    unittest.main()
