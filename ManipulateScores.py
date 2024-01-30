# This file summarizes functions to take lists with scores, manipulate these, and return the manipulated lists.
import copy
from evaluation import ce_score_fct
from generate_graphs import get_gnn_outs


class ManipulateScores:
    """
    Input: List of scores as in Beamsearch
    Functions: creating new lists with manipulated graphs for evaluating the Scores; Returning manipulated lists
    """

    def __init__(self, orig_scores, model, target_class) -> None:
        self._original_scores = orig_scores
        self._model = model
        self._target_class = target_class

    def score_wo_one_three_edges(self, scores):
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
                graph = self.delete_one_three_edges(graph)
                # recalculate Scores
                gnn_out = get_gnn_outs(graph, self._model, self._target_class)
                local_result['GNN_outs'].append(gnn_out)
            local_result['score'] = ce_score_fct(
                local_result['CE'], local_result['GNN_outs'], 0.5, 0, 'mean')

        new_scores = [round(g['score'], 2) for g in manipulated_results]
        print('old scores: ', old_scores)
        print('new scores: ', new_scores)
        return new_scores

    @staticmethod
    def delete_one_three_edges(graph_to_change):
        """
        Input: graph as ??
        Output: graph with all nodes that have one or three edges deleted
        """
        graph = copy.deepcopy(graph_to_change)
        print('graph before: ', graph)
        if ('1', 'to', '3') in graph.edge_types:
            del graph['1', 'to', '3']
        if ('3', 'to', '1') in graph.edge_types:
            del graph['3', 'to', '1']
        return graph

    def delete_zero_from_graph(self):
        pass

    def add_two_three_edge_to_start(self):
        pass
