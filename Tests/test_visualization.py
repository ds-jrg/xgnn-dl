import unittest
import torch
from torch_geometric.data import HeteroData
from visualization import visualize_hd


class TestVisualizeHD(unittest.TestCase):

    def test_visualize_hd(self):
        # Test case 1: Basic test case with no label to explain
        hd_graph = HeteroData()
        hd_graph['3'].x = torch.tensor([[1.], [1.]])

        hd_graph['2'].x = torch.tensor([[1.], [1.]])
        hd_graph['1'].x = torch.tensor([[1.], [1.]])
        hd_graph['0'].x = torch.tensor([[1.]])
        hd_graph[('3', 'to', '3')].edge_index = torch.tensor([[0, 1], [1, 0]])
        hd_graph[('3', 'to', '2')].edge_index = torch.tensor([[0, 0], [1, 0]])
        hd_graph[('2', 'to', '3')].edge_index = torch.tensor([[0, 1], [0, 0]])
        hd_graph[('1', 'to', '2')].edge_index = torch.tensor([[0, 1], [1, 0]])
        hd_graph[('2', 'to', '1')].edge_index = torch.tensor([[0, 1], [1, 0]])
        hd_graph[('2', 'to', '0')].edge_index = torch.tensor([[1], [0]])
        hd_graph[('0', 'to', '2')].edge_index = torch.tensor([[0], [1]])
        addname_for_save = "test_case_1"
        list_all_nodetypes = ["0", "1", "2", "3"]
        label_to_explain = "2"
        add_info = "Test case 1"
        name_folder = ""
        visualize_hd(hd_graph, addname_for_save, list_all_nodetypes, label_to_explain, add_info, name_folder)
        # Add assertions to check if the visualization is correct

        addname_for_save = "test_case_2"
        list_all_nodetypes = ["0", "1", "2", "3"]
        label_to_explain = "2"
        add_info = "Test case 2"
        name_folder = ""
        visualize_hd(hd_graph, addname_for_save, list_all_nodetypes, label_to_explain, add_info, name_folder)
        # Add assertions to check if the visualization is correct

        addname_for_save = "test_case_3"
        list_all_nodetypes = ["0", "1", "2", "3"]
        label_to_explain = "2"
        label_to_explain = None
        add_info = "Test case 3"
        name_folder = "custom_folder/"
        visualize_hd(hd_graph, addname_for_save, list_all_nodetypes, label_to_explain, add_info, name_folder)
        # Add assertions to check if the visualization is correct


if __name__ == '__main__':
    unittest.main()
