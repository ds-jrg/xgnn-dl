import unittest
from ManipulateScores import ManipulateScores
from torch_geometric.data import HeteroData, Data
import torch
import copy


class TestManipulateScores(unittest.TestCase):

    def setUp(self):
        # Set up the necessary objects for testing
        hdata = HeteroData()
        hdata['1'].x = torch.randn(3, 16)
        hdata['2'].x = torch.randn(3, 16)
        hdata['3'].x = torch.randn(3, 16)
        hdata['1', 'to', '2'].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        hdata['2', 'to', '1'].edge_index = torch.tensor([[1, 2, 0], [0, 1, 2]])
        hdata['2', 'to', '3'].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        hdata['3', 'to', '2'].edge_index = torch.tensor([[1, 2, 0], [0, 1, 2]])
        hdata['3', 'to', '1'].edge_index = torch.tensor([[0, 1, 2, 1, 0], [1, 2, 0, 0, 0]])
        hdata['1', 'to', '3'].edge_index = torch.tensor([[1, 2, 0, 0, 0], [0, 1, 2, 1, 0]])
        self.hdata = hdata

    def test_score_wo_one_three_edges(self):
        # Test the score_wo_one_three_edges method
        manipulate_scores = ManipulateScores()
        #new_scores = manipulate_scores.score_wo_one_three_edges(self.scores)

        # Assertions to check if the new scores are correct
        #self.assertEqual(len(new_scores), len(self.scores))
        # Add more assertions as needed

    def test_delete_one_three_edges(self):
        # Test the delete_one_three_edges method
        manipulate_scores = ManipulateScores()
        graph_to_change = self.hdata
        # new_graph = manipulate_scores.delete_one_three_edges(graph_to_change)

        # Assertions to check if the new graph is correct
        # Add assertions to check if the edges are deleted correctly

    def test_change_2_1_3_to_2_1_1(self):
        # Test the change_2_1_3_to_2_1_1 method
        manipulate_scores = ManipulateScores()
        graph_to_change = self.hdata
        # new_graph = manipulate_scores.change_2_1_3_to_2_1_1(graph_to_change)

        # Assertions to check if the new graph is correct
        # Add assertions to check if the edges are changed correctly

    def test_add_3_1_edges(self):
        # Test the add_3_1_edges method
        
            
        manipulate_scores = ManipulateScores()
        graph_to_change = self.hdata
        new_graph = manipulate_scores.add_3_1_edges(graph_to_change)

        # Assertions to check if the new graph is correct
        # Add assertions to check if the edges are added correctly

    def test_util_add_edge_to_graph(self):
        # Test the util_add_edge_to_graph method
        manipulate_scores = ManipulateScores()
        graph = self.hdata
        graph = copy.deepcopy(graph)
        nodetype1 = '1'
        edgetype = 'to'
        nodetype2 = '3'
        index1 = 0
        index2 = 0
        try:
            del graph['3', 'to', '1']
            del graph['1', 'to', '3']
        except Exception:
            pass
        new_graph = manipulate_scores.util_add_edge_to_graph(graph, nodetype1, edgetype, nodetype2, index1, index2)
        assert ('3', 'to', '1') in new_graph.edge_types, new_graph.edge_types
        assert ('1', 'to', '3') in new_graph.edge_types, new_graph.edge_types

        # Assertions to check if the new graph is correct
        # Add assertions to check if the edge is added correctly

    def test_util_get_neighbors_of_type(self):
        # Test the util_get_neighbors_of_type method
        manipulate_scores = ManipulateScores()
        graph = self.hdata
        nodetype_start = '1'
        nodetype_end = '3'
        index_start = 0
        neighbors = manipulate_scores.util_get_neighbors_of_type(graph, nodetype_start, nodetype_end, index_start)
        self.assertEqual(len(neighbors), 3)
        # Assertions to check if the neighbors are correct
        # Add assertions to check if the indices are correct


if __name__ == '__main__':
    unittest.main()
