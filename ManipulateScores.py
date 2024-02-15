# This file summarizes functions to take lists with scores, manipulate these, and return the manipulated lists.
import copy
import random
import torch
from evaluation import ce_score_fct
from generate_graphs import get_gnn_outs


class ManipulateScores:
    """
    Input: List of scores as in Beamsearch
    Functions: creating new lists with manipulated graphs for evaluating the Scores; Returning manipulated lists
    """

    def __init__(self, orig_scores=None, model=None, target_class=None, aggregation='max') -> None:
        self._original_scores = orig_scores
        self._model = model
        self._target_class = target_class
        self._aggregation = aggregation

    def score_manipulated_graphs(self, scores, new_graphs: list):
        """
        Scores are lists of dictionaries with the following keys: 'CE', 'graphs', 'GNN_outs', 'score', sorted by scores
        Input: List of scores as in Beamsearch
        Output: List of scores with all graphs that have one or three edges deleted
        """
        manipulated_results = copy.deepcopy(scores)
        old_scores = [round(g['score'], 2) for g in scores]
        # calculate old scores with regularization = 0; ensuring that only the GNN is taken into account

        for local_result in manipulated_results:
            local_result['GNN_outs'] = list()
            local_result['old_GNN_outs'] = list()
            for graph_old, graph_new in zip(new_graphs, local_result['graphs']):
                gnn_out_old = get_gnn_outs(graph_old, self._model, self._target_class)
                local_result['old_GNN_outs'].append(gnn_out_old)
                # recalculate Scores
                gnn_out = get_gnn_outs(graph_new, self._model, self._target_class)
                local_result['GNN_outs'].append(gnn_out)
            local_result['old_score'] = ce_score_fct(
                local_result['CE'], local_result['old_GNN_outs'], 0.0, 0, self._aggregation)
            local_result['score'] = ce_score_fct(
                local_result['CE'], local_result['GNN_outs'], 0.0, 0, self._aggregation)

        old_scores = [round(g['old_score'], 2) for g in manipulated_results]
        new_scores = [round(g['score'], 2) for g in manipulated_results]
        print('We have now omitted the regularization term for identifying the best graphs. ')
        print('Now scores with and without edges between top and bottom nodes are presented: ')
        print('old scores: ', old_scores)
        print('new scores: ', new_scores)
        return new_scores

    def score_wo_one_three_edges(self, scores):
        """
        Scores are lists of dictionaries with the following keys: 'CE', 'graphs', 'GNN_outs', 'score', sorted by scores

        Input: List of scores as in Beamsearch
        Output: List of scores with all graphs that have one or three edges deleted
        """
        manipulated_results = copy.deepcopy(scores)
        old_scores = [round(g['score'], 2) for g in scores]
        # calculate old scores with regularization = 0; ensuring that only the GNN is taken into account

        for local_result in manipulated_results:
            local_result['GNN_outs'] = list()
            local_result['old_GNN_outs'] = list()
            for graph in local_result['graphs']:
                gnn_out_old = get_gnn_outs(graph, self._model, self._target_class)
                local_result['old_GNN_outs'].append(gnn_out_old)
                graph = self.delete_one_three_edges(graph)
                # recalculate Scores
                gnn_out = get_gnn_outs(graph, self._model, self._target_class)
                local_result['GNN_outs'].append(gnn_out)
            local_result['old_score'] = ce_score_fct(
                local_result['CE'], local_result['old_GNN_outs'], 0.0, 0, self._aggregation)
            local_result['score'] = ce_score_fct(
                local_result['CE'], local_result['GNN_outs'], 0.0, 0, self._aggregation)

        old_scores = [round(g['old_score'], 2) for g in manipulated_results]
        new_scores = [round(g['score'], 2) for g in manipulated_results]
        print('We have now omitted the regularization term for identifying the best graphs. ')
        print('Now scores with and without edges between top and bottom nodes are presented: ')
        print('old scores: ', old_scores)
        print('new scores: ', new_scores)
        return new_scores

    def score_add_3_1_edges(self, scores):
        """
        Scores are lists of dictionaries with the following keys: 'CE', 'graphs', 'GNN_outs', 'score', sorted by scores

        Input: List of scores as in Beamsearch
        Output: List of scores with all graphs that have one or three edges deleted
        """
        manipulated_results = copy.deepcopy(scores)
        old_scores = [round(g['score'], 2) for g in scores]

        for local_result in manipulated_results:
            local_result['GNN_outs'] = list()
            for graph in local_result['graphs']:
                graph = self.add_3_1_edges(graph)
                # recalculate Scores
                gnn_out = get_gnn_outs(graph, self._model, self._target_class)
                local_result['GNN_outs'].append(gnn_out)
            local_result['score'] = ce_score_fct(
                local_result['CE'], local_result['GNN_outs'], 0.0, 0, 'mean')

        new_scores = [round(g['score'], 2) for g in manipulated_results]
        print('Now scores with additional 3-1 edges: ')
        print('old scores: ', old_scores)
        print('new scores: ', new_scores)
        return new_scores

    @staticmethod
    def delete_list_of_graphs_1_3(list_graphs):
        """
        Input: list of graphs as hdata
        Output: list of graphs with all nodes that have one or three edges deleted
        """
        new_list = []
        for graph in list_graphs:
            new_list.append(ManipulateScores.delete_one_three_edges(graph))
        return new_list

    @staticmethod
    def delete_one_three_edges(graph_to_change):
        """
        Input: graph as hdata
        Output: graph with all nodes that have one or three edges deleted
        """
        graph = copy.deepcopy(graph_to_change)
        if ('1', 'to', '3') in graph.edge_types:
            del graph['1', 'to', '3']
        if ('3', 'to', '1') in graph.edge_types:
            del graph['3', 'to', '1']
        return graph

    @staticmethod
    def change_2_1_3_to_2_1_1(graph_to_change):
        # TODO: Implement this method
        """
        Input: graph as hdata
        Output: Graph with 1-3 edges where the one is connected to a 2 get changed into 1-1 edges. 

        """
        graph = copy.deepcopy(graph_to_change)
        pass

    @staticmethod
    def list_of_graphs_add_3_1_edges(graphs_to_change):
        """
        Input: graphs_to_change as hdata
        Output: List of graphs where all 3s have an edge to a 1 and the other way around
        """
        result_list = []
        for graph in graphs_to_change:
            result_list.append(ManipulateScores.add_3_1_edges(graph))
        return result_list

    @staticmethod
    def add_3_1_edges(graph_to_change):
        """
        Input: graph as hdata
        Output: Graph where all 3s have an edge to a 1 and the other way around

        """
        graph = copy.deepcopy(graph_to_change)
        # 1. check all nodes of type 3
        num_nodes_type_3 = graph['3'].x.shape[0]
        num_nodes_type_1 = graph['1'].x.shape[0]
        for i in range(num_nodes_type_3):
            # check, if this node has a connection to nodetype 1
            indices_of_connected_one_nodes = ManipulateScores.util_get_neighbors_of_type(
                graph, '3', '1', i)
            if len(indices_of_connected_one_nodes) == 0:
                index_of_node_1 = random.randint(0, num_nodes_type_1 - 1)
                ManipulateScores.util_add_edge_to_graph(
                    graph, '3', 'to', '1', i, index_of_node_1)
        for i in range(num_nodes_type_1):
            # check, if this node has a connection to nodetype 3
            indices_of_connected_three_nodes = ManipulateScores.util_get_neighbors_of_type(
                graph, '1', '3', i)
            if len(indices_of_connected_three_nodes) == 0:
                index_of_node_3 = random.randint(0, num_nodes_type_3 - 1)
                ManipulateScores.util_add_edge_to_graph(
                    graph, '1', 'to', '3', i, index_of_node_3)
        return graph

    def delete_zero_from_graph(self):
        pass

    def add_two_three_edge_to_start(self):
        pass

    @staticmethod
    def util_add_edge_to_graph(graph, nodetype1, edgetype, nodetype2, index1, index2):
        # First: Check, if it is already there
        indicies_neighbors = ManipulateScores.util_get_neighbors_of_type(
            graph, nodetype1, nodetype2, index1)
        if index2 in indicies_neighbors:
            return graph
        if (nodetype1, edgetype, nodetype2) in graph.edge_types:
            graph[nodetype1, edgetype, nodetype2].edge_index = torch.cat(
                (graph[nodetype1, edgetype, nodetype2].edge_index, torch.tensor([[index1], [index2]])), 1)
            graph[nodetype2, 'to', nodetype1].edge_index = torch.cat(
                (graph[nodetype2, 'to', nodetype1].edge_index, torch.tensor([[index2], [index1]])), 1)
        else:
            graph[nodetype1, edgetype, nodetype2].edge_index = torch.tensor(
                [[index1], [index2]], dtype=torch.long)
            graph[nodetype2, 'to', nodetype1].edge_index = torch.tensor(
                [[index2], [index1]], dtype=torch.long)
        return graph

    @staticmethod
    def util_get_neighbors_of_type(graph, nodetype_start, nodetype_end, index_start) -> list:
        """
        Input: graph, nodetype_start, nodetype_end, index_start
        Output: list of indices of nodes of type nodetype_end that are connected to the node of type nodetype_start
        """
        if (nodetype_start, 'to', nodetype_end) not in graph.edge_types:
            return []
        edge_index = graph[nodetype_start, 'to', nodetype_end].edge_index
        indices_of_start_nodes = [i for i, node in enumerate(
            edge_index[0].tolist()) if node == index_start]
        connected_nodes = [edge_index[1].tolist()[i]
                           for i in indices_of_start_nodes]
        return connected_nodes
