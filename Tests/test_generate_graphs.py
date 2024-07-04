import unittest
from XLitOnto.XLit.generate_graphs import get_gnn_outs
from datasets import HeteroBAMotifDataset, GenerateRandomGraph, GraphMotifAugmenter
from models import HeteroGNNModel, HeteroGNNTrainer
import networkx as nx
import torch
from torch_geometric.data import HeteroData



class TestGetGnnOuts(unittest.TestCase):
    # TODO: datasets löschen; möglichst einfache Daten nehmen

    def setUp(self):
        type_to_classify = '3'
        self.type_to_classify = type_to_classify

        def test_new_datasets(num_nodes, num_motifs, test=False):
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
            # test: build small graph
            if test:
                g = nx.DiGraph()
                g.add_node(0, label='0')
                g.add_node(1, label='1')
                g.add_node(2, label='2')
                g.add_node(3, label='3')
                g.add_edge(0, 1)
                g.add_edge(1, 2)
                g.add_edge(2, 3)
                g.add_edge(3, 0)
                g.add_edge(1, 0)
                g.add_edge(2, 1)
                g.add_edge(3, 2)
                g.add_edge(0, 3)
                synthetic_graph = g

            # end test
            print('Printing manipulated synthetic graph:')
            print(synthetic_graph)
            dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
            if test:
                print('Printing manipulated synthetic graph:')
                print(synthetic_graph)
                # exit()
            if test:
                percentage_change = 0
            else:
                percentage_change = 0.4
            synthetic_graph = dataset_class._add_random_types(labels, '0', percentage_change)
            labels = [str(node['label']) for _, node in synthetic_graph.nodes(data=True) if 'label' in node]
            for i in range(0, 4):
                assert str(i) in labels, 'Label ' + str(i) + ' not in labels ' + str(labels)
            dataset = dataset_class._convert_labels_to_node_types()
            print('Data the GNN is trained on: ', dataset, dataset.x_dict)
            return dataset
        out_of_sample_data = test_new_datasets(3, 1, True)
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
            # TODO: Test if training-Dataset is undirected
            # TODO: Use pyg heterodata.is_undirected() -> bool method
            # TODO: Check, whether the test-dataset is undirected (similarly to trainig)
            # assert tests just for this one dataset
            assert self.hd[self.type_to_classify].x is not None, 'x_dict[self.type_to_classify] is None'
            assert self.hd.x_dict[self.type_to_classify].shape[1] == 1, 'x_dict[3] has shape ' + \
                str(self.hd.x_dict[self.type_to_classify].shape)
            print('Printing x_dict and edge_index_dict:')
            print(self.hd.x_dict)  # ,
            print(self.hd.edge_index_dict)

            data = HeteroData()
            data['3'].x = torch.tensor([[1.], [1.]])
            data[('3', 'to', '3')].edge_index = torch.tensor([[0, 1], [1, 0]])
            data['2'].x = torch.tensor([[1.], [1.]])
            #
            data[('3', 'to', '2')].edge_index = torch.tensor([[0, 1], [1, 0]])
            data[('2', 'to', '3')].edge_index = torch.tensor([[0, 1], [1, 0]])
            print('Test-data is directed?', data.is_directed())
            out = self.model(data.x_dict, data.edge_index_dict)

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
