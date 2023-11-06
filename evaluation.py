# Here, some scoring functions and other evaluation functions are implemented

# ----------- evaluating class expressions: Currently not in use. ------------
import random as random
import os.path as osp
from torch_geometric.data import HeteroData
import torch
import dgl
import sys
import copy
from ce_generation import generate_cedict_from_ce
from create_random_ce import length_ce, remove_front, find_class, count_classes
from graph_generation import compute_prediction_ce
import pandas as pd

from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom, OWLObjectUnionOf
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import IRI
from owlapy.render import DLSyntaxObjectRenderer
dlsr = DLSyntaxObjectRenderer()


def available_edges_with_nodeid(graph, current_type, current_id, edgetype='to'):
    # graph is in dictionary form
    list_result = list()
    for key, value in graph.items():
        if key[0] == current_type and key[1] == edgetype:
            for _, indexvalue in enumerate(value[0].tolist()):
                if current_id == indexvalue:
                    list_result.append([key[2], value[1].tolist()[_], value[2][_]])
    return list_result


def ce_fidelity(ce_for_fid, modelfid, datasetfid, node_type_expl, label_expl=-1, random_seed=1):
    fid_result = -1
    mask = datasetfid[node_type_expl].test_mask
    mask_tf = 0
    for value in mask.tolist():
        if str(value) == 'True' or str(value) == 'False':
            mask_tf = 1
            break
    metagraph = datasetfid.to_dict()
    # falls node_type_expl == -1: Ändere dies auf das letzte aller möglichen labels
    if label_expl == -1:
        list_labels = datasetfid[node_type_expl].y
        label_expl = max(set(list_labels))
    modelfid.eval()
    pred = modelfid(datasetfid.x_dict, datasetfid.edge_index_dict).argmax(dim=-1)
    pred_list = pred.tolist()
    for index, value in enumerate(pred_list):
        if value != label_expl:
            pred_list[index] = 0
        else:
            pred_list[index] = 1
    pred = torch.tensor(pred_list)
    if mask_tf == 0:
        mask = datasetfid[node_type_expl]['test_mask']
        cedict = generate_cedict_from_ce(ce_for_fid)
        # mask = select_ones(mask, 100)
        # create new vector with samples only as true vector
        smaller_mask = random.sample(mask.tolist(), k=min(200, len(mask.tolist())))
        mask = torch.tensor(smaller_mask)
    else:
        indices_of_ones = [i for i, value in enumerate(mask.tolist()) if value == True]
        chosen_indices = random.sample(indices_of_ones, k=min(20, len(indices_of_ones)))
        mask = [i if i in chosen_indices else 0 for i in range(len(mask.tolist()))]
        mask = [x for x in mask if x != 0]
        mask = torch.tensor(mask)
        sys.exit()
    count_fids = 0
    count_zeros_test = 0
    count_zeros_gnn = 0
    for index in mask.tolist():
        cedict = generate_cedict_from_ce(ce_for_fid)
        result_ce_fid = compute_prediction_ce(cedict, metagraph, node_type_expl, index)
        if pred[index] == result_ce_fid:
            count_fids += 1
        if result_ce_fid == 0:
            count_zeros_test += 1
        if pred[index] == 0:
            count_zeros_gnn += 1
    fid_result = round(float(count_fids) / float(len(mask.tolist())), 2)
    return fid_result


# TODO: Think of cases, where this could not work: How would we find the edge 1-1 in the house, if there are no 2-3 edges ?
# current_graph_node is of form ['3',0], ['2',0]. ['2',1], etc. [nodetype, nodeid_of_nodetype]
def ce_confusion_iterative(ce, graph, current_graph_node):
    """
    This function takes in an OWL class expression (ce), a graph, and a current graph node and returns a set of edges, a boolean value, and a current graph node. 

    Parameters:
    ce (OWLClass): An OWL class expression.
    graph (Graph): A graph.
    current_graph_node (list): A list containing the current graph node.

    Returns:
    result (set): A set of edges.
    return_truth (bool): A boolean value indicating whether the function returned a valid result.
    current_graph_node (list): A list containing the current graph node.
    """
    # TODO: Insert 'abstract current nodes', if a edge to a node not in the graph or without specified nodetype is called
    # save the number of abstract edges used (later, maybe instead of True / False as feedback ?

    result = set()
    if isinstance(ce, OWLClass):
        if current_graph_node[0] != remove_front(ce.to_string_id()):
            return result, False, current_graph_node
        else:
            return result, True, current_graph_node
    elif isinstance(ce, OWLObjectProperty):
        edgdetype = remove_front(ce.to_string_id())
        # form should be [edge, endnodetype, endnodeid]
        available_edges = available_edges_with_nodeid(graph, current_graph_node[0], current_graph_node[1], edgetype)
        if len(available_edges) > 0:
            # TODO: Add this edge to result, as the edge has been found
            # result.update()
            # retrieve all available edges
            set_possible_edges = set()
            for aved in available_edges:
                set_possible_edges.update(aved[2])
            for edgeind in set_possible_edges:
                if edgeind not in result:
                    result.update(edgeind)
                    break
            return result, True, current_graph_node
        return result, False, current_graph_node
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        if isinstance(ce._property, list):
            edgetype = remove_front(ce._property[0].to_string_id())
        else:
            edgetype = remove_front(ce._property.to_string_id())
        # form should be [edge, endnodetype, endnodeid]
        available_edges = available_edges_with_nodeid(graph, current_graph_node[0], current_graph_node[1], edgetype)
        current_best_length = len(result)
        result_copy = copy.deepcopy(result)
        local_result = set()
        local_current_grnd = current_graph_node
        some_edgewas_true = False
        for aved in available_edges:
            local_result = set()
            for i in result_copy:
                local_result.update(set(i))
            local_result.add(aved[2])
            feed1, feed2, current_graph_node = ce_confusion_iterative(ce._filler, graph, [aved[0], aved[1]])
            if feed2:
                some_edgewas_true = True
                # (319, feed1, current_graph_node)
                current_best_length = len(feed1)
                local_result_intern = feed1
                local_current_grnd = current_graph_node
                return set(list(local_result)+list(local_result_intern)), True, local_current_grnd
        if some_edgewas_true == False:
            current_graph_node = 'abstract'
            return result, True, current_graph_node

        # if current_best_length >0:
        #    return set(list(local_result)+list(local_result_intern)), True, local_current_grnd
        # else:
        #    return result, False, current_graph_node
    elif isinstance(ce, OWLObjectIntersectionOf):
        return_truth = True
        for op in ce.operands():  # TODO: First select class if available, then further edges
            if isinstance(op, OWLClass) == True:
                feed1, feed2, current_graph_node = ce_confusion_iterative(op, graph, current_graph_node)
                if feed1 != None:
                    result.update(list(feed1))
                if feed2 == False:
                    return_truth = False
        for op in ce.operands():
            if isinstance(op, OWLClass) == False:
                feed1, feed2, current_graph_node = ce_confusion_iterative(op, graph, current_graph_node)
                if feed1 != None:
                    result.update(list(feed1))
                if feed2 == False:
                    return_truth = False
        return result, return_truth, current_graph_node
    else:
        return result, False, current_graph_node
    return result, False, current_graph_node


def ce_confusion(ce,  motif='house'):
    motifgraph = dict()
    if motif == 'house':
        motifgraph = {('3', 'to', '2'): (torch.tensor([0, 0], dtype=torch.long),
                                         torch.tensor([0, 1], dtype=torch.long), [0, 1]),
                      ('2', 'to', '3'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([0, 0], dtype=torch.long), [1, 0]),
                      ('2', 'to', '1'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([0, 1], dtype=torch.long), [2, 3]),
                      ('1', 'to', '2'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([0, 1], dtype=torch.long), [3, 2]),
                      ('2', 'to', '2'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([1, 0], dtype=torch.long), [4, 4]),
                      ('1', 'to', '1'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([1, 0], dtype=torch.long), [5, 5]),
                      # now the abstract class is included
                      ('0', 'to', 'abstract'): (torch.tensor([0], dtype=torch.long),
                                                torch.tensor([0], dtype=torch.long), [-1]),
                      ('abstract', 'to', '0'): (torch.tensor([0], dtype=torch.long),
                                                torch.tensor([0], dtype=torch.long), [-1]),
                      ('1', 'to', 'abstract'): (torch.tensor([0, 1], dtype=torch.long),
                                                torch.tensor([0, 0], dtype=torch.long), [-1, -1]),
                      ('abstract', 'to', '1'): (torch.tensor([0, 0], dtype=torch.long),
                                                torch.tensor([0, 1], dtype=torch.long), [-1, -1]),
                      ('2', 'to', 'abstract'): (torch.tensor([0, 1], dtype=torch.long),
                                                torch.tensor([0, 0], dtype=torch.long), [-1, -1]),
                      ('abstract', 'to', '2'): (torch.tensor([0, 0], dtype=torch.long),
                                                torch.tensor([0, 1], dtype=torch.long), [-1, -1]),
                      ('3', 'to', 'abstract'): (torch.tensor([0], dtype=torch.long),
                                                torch.tensor([0], dtype=torch.long), [-1]),
                      ('abstract', 'to', '3'): (torch.tensor([0], dtype=torch.long),
                                                torch.tensor([0], dtype=torch.long), [-1]),
                      ('abstract', 'to', 'abstract'): (torch.tensor([0], dtype=torch.long),
                                                       torch.tensor([0], dtype=torch.long), [-1])
                      }
    test_bla = ce_confusion_iterative(ce, motifgraph, ['3', 0])
    # print(test_bla)


def ce_score_fct(ce, list_gnn_outs, lambdaone, lambdatwo):
    # avg_gnn_outs-lambda len_ce - lambda_var
    length_of_ce = length_ce(ce)
    mean = sum(list_gnn_outs) / len(list_gnn_outs)
    squared_diffs = [(x - mean) ** 2 for x in list_gnn_outs]
    sum_squared_diffs = sum(squared_diffs)
    variance = sum_squared_diffs / (len(list_gnn_outs))
    return mean-lambdaone*length_of_ce-lambdatwo*variance


def get_accuracy_baheteroshapes(ce):
    # Implementation following soon!
    pass


# ------------------- new functions ------------------- (10.2023)
def find_adjacent_edges(hetero_data: HeteroData, node_type: str, node_id: int):
    """
    Find adjacent edges for a specific node in a HeteroData object.

    Parameters:
    - hetero_data (HeteroData): The HeteroData object containing the graph data.
    - node_type (int): The type of node for which to find adjacent edges.
    - node_id (int): The ID of the node for which to find adjacent edges.

    Returns:
    - list: A list of tuples representing adjacent edges. Each tuple is of the form (source_node, target_node, edge_type).
    """
    adjacent_edges = []
    # check, if heter_data is valid:
    if isinstance(hetero_data, dict):
        hetero_data = HeteroData(hetero_data)
    if not isinstance(hetero_data, HeteroData):
        raise TypeError('hetero_data must be of type HeteroData')
    # Iterate through all possible edge types in the HeteroData object
    for edge_type in hetero_data.edge_types:
        edge_data = hetero_data[edge_type]['edge_index']
        mask = edge_data[0] == node_id
        src_type, rel_type, dst_type = edge_type
        if src_type == node_type:
            target_nodes = edge_data[1][mask]
            for target_node_id in target_nodes:
                adjacent_edges.append((target_node_id.item(), dst_type, edge_type))
            # Check for edges terminating at the given node
    # return dict of adjacent edges: {(str, str, str) : tensor[(tensor, tensor)]}
    adjacent_edges = set(adjacent_edges)
    return adjacent_edges


def ce_fast_instance_checker(ce, dataset, current_node_type, current_id):
    """
    This function gives back the set of instances in the dataset, for which the CE holds true. 
    Input:
    ce: OWL class expression
    dataset: dataset in Pytorch Geometric format;
    current_node_type: node type of the current node; called with the node-type to be explained
    current_id: The current id of the node which is checked for validity

    Output:
    the set of nodes in the graph, where the CE "ends"

    Assumptions:
    - The dataset is a heterogenous graph
    - The dataset contains only one graph
    - Each node has exactly one type
    - Each individual in a CE has exactly one type
    """
    # TODO: change "union" to intersection -> then, if somewhere set() is returned
    # Then, the overall result is also set() and if not, then the overal result is only the node we started the search
    valid_adjacent_nodes = set()
    if isinstance(ce, OWLClass):
        if remove_front(ce.to_string_id()) == current_node_type:
            return set([(current_node_type, current_id)])
        else:
            return set()
    elif isinstance(ce, OWLObjectIntersectionOf):
        top_class = find_class(ce)
        #  normalize CE-classes and node_type_expl to one format
        top_class = remove_front(top_class.to_string_id())
        if top_class != current_node_type:
            return set()

        for op in ce.operands():
            if not isinstance(op, OWLClass):
                valid_adjacent_nodes.update(ce_fast_instance_checker(op, dataset, current_node_type, current_id))
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        new_class = find_class(ce._filler)
        new_class = remove_front(new_class.to_string_id())
        adjacent_edges = find_adjacent_edges(dataset, current_node_type, current_id)
        for edge in adjacent_edges:
            if edge[1] == new_class:
                pass
                # valid_adjacent_nodes.add(edge)
        for edge in adjacent_edges:
            new_adjacent_nodes = ce_fast_instance_checker(ce._filler, dataset, edge[1], edge[0])
            if not new_adjacent_nodes:
                pass
            else:
                valid_adjacent_nodes.update(new_adjacent_nodes)
    return valid_adjacent_nodes


def fidelity_el(ce, dataset, node_type_to_expl, model, label_to_expl):
    # find all ids of the test data of dataset
    count = 0
    if hasattr(node_type_to_expl, 'to_string_id'):
        node_type_to_expl = remove_front(node_type_to_expl.to_string_id())
    mask = dataset[node_type_to_expl].test_mask
    mask_tf = 0
    # check, if 0/1 or True/False is used as mask
    for value in mask.tolist():
        if str(value) == 'True' or str(value) == 'False':
            mask_tf = 1
            break
    if mask_tf == 1:
        indices_of_ones = [i for i, value in enumerate(mask.tolist()) if value == True]
        chosen_indices = random.sample(indices_of_ones, k=min(20, len(indices_of_ones)))
        mask = [i if i in chosen_indices else 0 for i in range(len(mask.tolist()))]
        mask = [x for x in mask if x != 0]
        mask = torch.tensor(mask)
    model.eval()
    pred = model(dataset.x_dict, dataset.edge_index_dict).argmax(dim=-1)
    pred_list = pred.tolist()
    for index, value in enumerate(pred_list):
        if value != label_to_expl:
            pred_list[index] = 0
        else:
            pred_list[index] = 1
    for id in mask.tolist():
        current_id = id
        return_id = 1
        return_set = ce_fast_instance_checker(ce, dataset, node_type_to_expl, current_id)
        return_gnn = pred_list[id]  # get instead the gnn feedback
        if not return_set:
            return_id = 0
        if return_id == return_gnn:
            count += 1
    fid_result = round(float(count) / float(len(mask.tolist())), 2)
    return fid_result


class Accuracy_El:
    def __init__(self):
        dict_motif = {('3', 'to', '2'): (torch.tensor([0, 0], dtype=torch.long),
                                         torch.tensor([0, 1], dtype=torch.long), [0, 1]),
                      ('2', 'to', '3'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([0, 0], dtype=torch.long), [1, 0]),
                      ('2', 'to', '1'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([0, 1], dtype=torch.long), [2, 3]),
                      ('1', 'to', '2'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([0, 1], dtype=torch.long), [3, 2]),
                      ('2', 'to', '2'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([1, 0], dtype=torch.long), [4, 4]),
                      ('1', 'to', '1'): (torch.tensor([0, 1], dtype=torch.long),
                                         torch.tensor([1, 0], dtype=torch.long), [5, 5]),
                      # now the abstract class is included
                      ('0', 'to', 'abstract'): (torch.tensor([0], dtype=torch.long),
                                                torch.tensor([0], dtype=torch.long), [-1]),
                      ('abstract', 'to', '0'): (torch.tensor([0], dtype=torch.long),
                                                torch.tensor([0], dtype=torch.long), [-1]),
                      ('1', 'to', 'abstract'): (torch.tensor([0, 1], dtype=torch.long),
                                                torch.tensor([0, 0], dtype=torch.long), [-1, -1]),
                      ('abstract', 'to', '1'): (torch.tensor([0, 0], dtype=torch.long),
                                                torch.tensor([0, 1], dtype=torch.long), [-1, -1]),
                      ('2', 'to', 'abstract'): (torch.tensor([0, 1], dtype=torch.long),
                                                torch.tensor([0, 0], dtype=torch.long), [-1, -1]),
                      ('abstract', 'to', '2'): (torch.tensor([0, 0], dtype=torch.long),
                                                torch.tensor([0, 1], dtype=torch.long), [-1, -1]),
                      ('3', 'to', 'abstract'): (torch.tensor([0], dtype=torch.long),
                                                torch.tensor([0], dtype=torch.long), [-1]),
                      ('abstract', 'to', '3'): (torch.tensor([0], dtype=torch.long),
                                                torch.tensor([0], dtype=torch.long), [-1]),
                      ('abstract', 'to', 'abstract'): (torch.tensor([0], dtype=torch.long),
                                                       torch.tensor([0], dtype=torch.long), [-1])
                      }
        heterodata_motif = HeteroData()
        for edge, indices in dict_motif.items():
            heterodata_motif[edge].edge_index = (indices[0], indices[1], torch.tensor(indices[2]))
        self.dict_motif = heterodata_motif
        self.list_results = list()
        dict_found_nodes = {('3', 0): False, ('2', 0): False, ('2', 1): False, ('1', 0): False, ('1', 1): False}
        number_of_false_positives = 0
        pos_graph = {'nodetype': '3', 'id': 0}
        dict_onepath = {'result': dict_found_nodes, 'position': pos_graph,
                        'fp': number_of_false_positives, 'accuracy': None, 'found_classes': 0, 'total_classes': -1}
        self.list_results.append(dict_onepath)
        self.current_path_index = 0

    def _calc_accuracy(self):
        """
        Calculate the accuracy of the current path
        """
        for dict in self.list_results:
            dict['accuracy'] = sum(1 for value in dict['result'].values() if value is True) / (5 + dict['fp'])
        print(self.list_results[0]['result'], self.list_results[0]['fp'], self.list_results[0]['accuracy'])

    def _ce_accuracy_iterate_house(self, ce):
        """
        pos_graph: dict nodetype : str, id: int
        dict_motif: dict of the motif for its graph structure
        dict_result: dict of the nodetypes : True or False (found or not found)
        """
        if isinstance(ce, OWLClass):
            # Question: Here what to do exactly ?? Should we check, if the class is in the graph ??
            # Should we update the current position here? -> no, that is not possible
            self.list_results[self.current_path_index]['found_classes'] += 1
            print(remove_front(ce.to_string_id()), self.list_results[self.current_path_index]['position'])
            if remove_front(ce.to_string_id()) == self.list_results[self.current_path_index]['position']['nodetype']:
                print(remove_front(ce.to_string_id()), self.list_results[self.current_path_index]['position'])
                self.list_results[self.current_path_index]['result'][(self.list_results[self.current_path_index]['position']['nodetype'],
                                                                      self.list_results[self.current_path_index]['position']['id'])] = True
            else:
                self.list_results[self.current_path_index]['fp'] += 1
        elif isinstance(ce, OWLObjectIntersectionOf):
            for op in ce.operands():
                self._ce_accuracy_iterate_house(op)
        elif isinstance(ce, OWLObjectSomeValuesFrom):
            new_class = remove_front(find_class(ce._filler).to_string_id())
            adjacent_edges = find_adjacent_edges(
                self.dict_motif, self.list_results[self.current_path_index]['position']['nodetype'], self.list_results[self.current_path_index]['position']['id'])
            new_abstract_index = 0
            # iterate somehow over all possible paths
            if len(adjacent_edges) == 1:
                # append also an edge to abstract node and test that one
                self.list_results.append(copy.deepcopy(self.list_results[self.current_path_index]))
                new_abstract_index = len(self.list_results) - 1
                self.list_results[-1]['position'] = {'nodetype': 'abstract', 'id': 0}
                edge = adjacent_edges[0]
                if edge[1] == new_class:
                    # update the current path
                    self.list_results[self.current_path_index]['position'] = {'nodetype': edge[1], 'id': edge[0]}
            elif len(adjacent_edges) == 0:
                self.list_results[self.current_path_index]['fp'] += 1
                self.list_results[self.current_path_index]['position'] = {'nodetype': 'abstract', 'id': 0}
                new_abstract_index = self.current_path_index
            else:
                # append also an edge to abstract node and test that one
                self.list_results.append(copy.deepcopy(self.list_results[self.current_path_index]))
                new_abstract_index = len(self.list_results) - 1
                count_current_path_index = 0
                adjacent_edges = list(set(adjacent_edges))
                current_index = copy.deepcopy(self.current_path_index)
                for edge in adjacent_edges:
                    if edge[1] == new_class:
                        self.list_results.append(copy.deepcopy(self.list_results[current_index]))
                        self.list_results[-1]['position'] = {'nodetype': edge[1], 'id': edge[0]}
                        self.current_path_index = len(self.list_results) -1
                        self._ce_accuracy_iterate_house(ce._filler)
                        count_current_path_index += 1
            self.list_results[new_abstract_index]['position'] = {'nodetype': 'abstract', 'id': 0}
            self.current_path_index = new_abstract_index
            self._ce_accuracy_iterate_house(ce._filler)

    def ce_accuracy_to_house(self, ce):
        """
        Idea: 
        For each class, save a CE to this class (in 3-2-2-1-1 style)
        Then, check for each CE, if it is valid in the house
        If not: change classes (except the last class) to 'abstract'
        Then: check again, and save all intermediate classes which are abstract as "loss"
            If this is done top-bottom, then we only need to check for the intermediate cases is the CE is not valid
        We then add up all the end-CE-results
        Simultaneously, we store the found nodes of the motif.
        """
        print('Evaluation of the CE: ', dlsr.render(ce))
        # Initialize the top class
        top_class = find_class(ce)
        if not remove_front(top_class.to_string_id()) == '3':
            self.list_results[self.current_path_index]['position']['nodetype'] = 'abstract'
            self.list_results[self.current_path_index]['fp'] += 1
        self._ce_accuracy_iterate_house(ce)
        self._calc_accuracy()
        sorted_list_results = sorted(
            self.list_results, key=lambda x: x['accuracy'] if x['accuracy'] is not None else float('-inf'), reverse=True)
        self.list_results.sort(key=lambda x: x['accuracy'] if x['accuracy'] is not None else float('-inf'), reverse=True)
        return sorted_list_results[0]['accuracy']
