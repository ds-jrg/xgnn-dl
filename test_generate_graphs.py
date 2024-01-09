import unittest
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

            # test if the dataset is created correctly
            labels = [str(node['label']) for _, node in synthetic_graph.nodes(data=True) if 'label' in node]
            for i in range(1, 4):
                assert str(i) in labels, 'Label ' + str(i) + ' not in labels ' + str(labels)
            dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
            synthetic_graph = dataset_class._add_random_types(labels, '0')
            labels = [str(node['label']) for _, node in synthetic_graph.nodes(data=True) if 'label' in node]
            for i in range(1, 4):
                assert str(i) in labels, 'Label ' + str(i) + ' not in labels ' + str(labels)
            dataset = dataset_class._convert_labels_to_node_types()
            print('Data the GNN is trained on: ', dataset, dataset.x_dict)
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
        try:
            out = self.model(self.hd.x_dict, self.hd.edge_index_dict)
            result = get_gnn_outs(self.hd, self.model, -1)
            assert isinstance(result, float)
        except AttributeError:
            # out = self.model(self.hd.x_dict, self.hd.edge_index_dict)
            assert False, ('Probably dim has not worked', self.hd, self.hd.x_dict)
            # Error: dim >= x.dim() or dim < -x.dim(): AttributeError: 'NoneType' object has no attribute 'dim'
            # has values like {'0': tensor([[1.], [1.]]), '2':


if __name__ == '__main__':
    unittest.main()
