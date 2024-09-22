# Here, some scoring functions and other evaluation functions are implemented

# ----------- evaluating class expressions: Currently not in use. ------------
from create_random_ce import CEUtils, Mutation
import random as random
import os.path as osp
from torch_geometric.data import HeteroData
import torch
import sys
import copy
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
dlsr = DLSyntaxObjectRenderer()


# --------- new code ------------
class InstanceChecker():
    """
    This class is used to find all instances, which fulfill a certain class expression
    Input: graph (Heterodata), CE (owlapy)
    Output: a list of labels for each node (saved as a list; index relates to index of node)
    Summary of functions:
    - fast_instance_checker_uic: Find all instances in the graph, which:
        have unions
        have intersections
        have cardinality restrictions (minimum cardinality)

    """

    def __init__(self, hdatagraph) -> None:
        self._graph = hdatagraph
    # TODO: Run in parallel

    def get_adjacent_nodes(self, current_nodetype, current_id, new_edgetype=None, new_nodetypes=None) -> dict:
        """
            gets all adjacent nodes of the current node, with only the nodetypes wanted.
            Input:
            - hdata: graph 
            - current_nodetype: node type of current node, where we want neighbors from
            - current_id: id of current node
            - new_edgetype: edge type of the edge we want to follow (if None, all edges are considered)
            - new_nodetypes: list of node types we want to follow / we are looking for.
                If None, then all nodes are considered. TODO: Implement this.
        """
        if new_edgetype is None:
            new_edgetype = True

        dict_adjacent_nodes = {}
        assert new_edgetype is not None
        assert new_nodetypes is not None
        for new_nodetype in new_nodetypes:
            dict_adjacent_nodes[new_nodetype] = []
        for edgetype in self._graph.edge_types:
            if edgetype[0] == current_nodetype and edgetype[1] == new_edgetype:
                if new_nodetypes is None:

                    for index, target_node in enumerate(self._graph[edgetype]['edge_index'][0]):
                        if target_node == current_id:
                            dict_adjacent_nodes[edgetype[2]].append(
                                self._graph[edgetype]['edge_index'][1][index].item())
                else:
                    if edgetype[2] in new_nodetypes:
                        for index, target_node in enumerate(self._graph[edgetype]['edge_index'][0]):
                            if target_node == current_id:
                                dict_adjacent_nodes[edgetype[2]].append(
                                    self._graph[edgetype]['edge_index'][1][index].item())
        return dict_adjacent_nodes

    def fast_instance_checker_uic(self, ce):
        """
        This function gives back the set of nodes in the dataset, 
        for which the CE holds true.
        Workflow:
        We start with all nodes and make this iteretively smaller.
        In the first iteration, we only check the nodetypes available
        Then, we check for all remaining nodes if the CE holds true.

        For each .. we do : ..
        intersection : We continue with all operands, all of them have to be satisfied
        union: We continue with all operands, at least one of them has to be satisfied
        cardinality : We continue with all operands of the new class

        Current node-types and node-ids for each inters. and union are saved:
        {node-type : [ids]}

        The result is stored as "result_instances and is a dictionary
        {node types: [node ids]}
        """
        hdata = self._graph
        check_instances = {}
        for node_type in hdata.node_types:
            check_instances[node_type] = set(
                range(hdata[node_type].num_nodes))

        def retrieve_top_classes(ce):
            if isinstance(ce, OWLClass):
                str_class = str(dlsr.render(ce))
                return [str_class]
            elif isinstance(ce, OWLObjectIntersectionOf):
                result = list()
                for op in ce.operands():
                    new_class = retrieve_top_classes(op)
                    if 0 < len(new_class):
                        result.extend(new_class)
                if len(result) >= 2:
                    return list()
                else:
                    return result
            elif isinstance(ce, OWLObjectUnionOf):
                result = list()
                for op in ce.operands():
                    result.append(op)
                return result
            return list()
        top_classes = retrieve_top_classes(ce)
        # check_instances are all instances
        # which have the correct top class as a nodetype
        for nodetype in hdata.node_types:
            if nodetype not in top_classes:
                check_instances.pop(nodetype, None)

        def iterate_through_graph(ce, current_nodetype, current_id):
            if isinstance(ce, OWLClass):
                if str(current_nodetype) == str(dlsr.render(ce)):
                    return True
                return False
            elif isinstance(ce, OWLObjectUnionOf):
                for op in ce.operands():
                    if iterate_through_graph(op, current_nodetype, current_id):
                        return True
                return False
            elif isinstance(ce, OWLObjectIntersectionOf):
                result = True
                for op in ce.operands():
                    result = result and iterate_through_graph(
                        op, current_nodetype, current_id)
                return result
            elif isinstance(ce, OWLObjectMinCardinality):
                # retrieve nodes that fit the edge type
                # then run this function
                # if counts of True higher or equal the cardinality
                # return True
                edgetype = str(dlsr.render(ce._property))
                new_nodetypes = retrieve_top_classes(ce._filler)
                dict_adjacent_nodes = self.get_adjacent_nodes(
                    current_nodetype, current_id, edgetype, new_nodetypes)
                # this is a dict: type : ids
                number_trues = 0
                for nodetype in dict_adjacent_nodes.keys():
                    for id in dict_adjacent_nodes[nodetype]:
                        if iterate_through_graph(ce._filler, current_nodetype=nodetype, current_id=id):
                            number_trues += 1
                            if number_trues >= ce._cardinality:
                                return True
                return False
        result_instances = dict()
        for nodetype in check_instances:
            for id in check_instances[nodetype]:
                if iterate_through_graph(ce, current_nodetype=nodetype, current_id=id):
                    result_instances.setdefault(
                        str(nodetype), []).append(id)
        return result_instances


class FidelityEvaluator():
    """
    This class is used to evaluate the fidelity of a given explanation.
    Input: graph (Heterodata), CE (owlapy), gnn (PyG)
    Output: fidelity score
    Summary of functions:
    - evaluate_fidelity: Evaluate the fidelity of a given explanation.
    """

    def __init__(self, hdata, gnn, type_to_explain=None):
        self.hdata = hdata
        self.gnn = gnn
        self.InstanceChecker = InstanceChecker(hdata)
        self.type_to_explain = type_to_explain
        result_gnn = self.gnn(
            self.hdata.x_dict, self.hdata.edge_index_dict)
        result_gnn = result_gnn.argmax(dim=-1)
        self.result_gnn = list(int(i) for i in result_gnn)

    def fidelity_tp_fp_tn_fn(self, ce):
        ce_instances = self.InstanceChecker.fast_instance_checker_uic(ce)
        # TODO: setting type_to_explain is wrong
        if self.type_to_explain is None:
            self.type_to_explain = CEUtils.return_nth_class(ce, 1)
            if isinstance(self.type_to_explain, OWLClassExpression):
                self.type_to_explain = str(dlsr.render(self.type_to_explain))
        if self.type_to_explain not in ce_instances.keys():
            return 0, 0, 0, 0
        list_indices_01 = [
            i in ce_instances[self.type_to_explain] for i in range(self.hdata[self.type_to_explain].num_nodes)]
        # compare to ground truth

        # get tp, fp, tn, fn)
        tp = sum(list_indices_01[i] and self.result_gnn[i]
                 for i in range(len(list_indices_01)))
        fp = sum(list_indices_01[i] and not self.result_gnn[i]
                 for i in range(len(list_indices_01)))
        tn = sum(not list_indices_01[i] and not self.result_gnn[i]
                 for i in range(len(list_indices_01)))
        fn = sum(not list_indices_01[i] and self.result_gnn[i]
                 for i in range(len(list_indices_01)))
        return tp, fp, tn, fn

    def score_fid_accuracy(self, ce):
        tp, fp, tn, fn = self.fidelity_tp_fp_tn_fn(ce)
        if tp + fp + tn + fn == 0:
            return 0
        return (tp + tn) / (tp + tn + fp + fn)
