from create_random_ce import CEUtils, Mutation
import torch
import torch_geometric
from torch_geometric.data import HeteroData
import copy
import random
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
from datasets import GraphLibraryConverter
dlsr = DLSyntaxObjectRenderer()


class PyGfromCE():
    """
    This class summarizes all functions, which are needed to create a PyG from a CE.
    ultimate function:  "create_pyg_from_ce"
    """

    def __init__(self) -> None:
        self.graph = HeteroData()
        self.list_edges = []
        self.prototype_edgetriple = {'start': None,
                                     'edge': None,
                                     'end': None,
                                     'ids': {'start': 0, 'end': 1}}
        self.list_current_edgetriples = []
        self.current_position = 'start'
        self.topce = None  # to store, if we are in union or intersection
        self.current_class = None
        self.is_in_filler = False

    @staticmethod
    def add_edge_to_graphdict(edge: dict, graphdict: dict):
        """
        1. get number of nodes of types to be added
        2. add to graphdict (which can be directly used to create a hdata PyG Graph)
        Create Graphs like:
        hetero_graph[(start_type, edge_type, end_type)].edge_index = torch.tensor(
                [list_ids_start, list_ids_end])
        """
        new_edge = (edge['start'], edge['edge'], edge['end'])
        if new_edge not in graphdict:
            graphdict[new_edge] = [[edge['ids']['start']], [
                                   edge['ids']['end']]]
        else:
            graphdict[new_edge][0].append([edge['ids']['start']])
            graphdict[new_edge][1].append([edge['ids']['end']])
        return graphdict

    def create_pyg_from_ce(self, ce):
        """
        Create a PyG from a CE.
        current_class states the class of this level in the CE and ensures, that one node does not have different classses.
        current_triples is the list of unfulfilled edges.
        list_edges is the list of edges for Heterodata.
        start_class is class for new edges
        current_position is, if classes are added to beginning or ending
        """
        new_ce = CEUtils.flatten_ce_class_first(ce)

        print('current ce: ', ce, 'new ce: ', new_ce)
        assert isinstance(new_ce, OWLClassExpression)

        def find_edges(ce, start_class=None, current_id=0):
            """
            All nary CEs have first classes and then further properties.
            """
            print('current ce: ', ce)
            assert isinstance(ce, OWLClassExpression)
            print('current list: ', self.list_current_edgetriples, self.list_edges)
            if isinstance(ce, OWLClass):
                if start_class is None:
                    start_class = ce
                    self.current_class = ce
                for triple in self.list_current_edgetriples:
                    if triple[self.current_position] is None:
                        if triple['ids'][self.current_position] is None:
                            triple['ids'][self.current_position] = current_id
                        triple[self.current_position] = CEUtils.get_name_from_class_or_property(
                            ce)
                self.current_position = 'start'
            elif isinstance(ce, OWLObjectCardinalityRestriction):
                assert (self.current_position == 'start')
                if start_class is None:
                    start_class = self.current_class
                for i in range(ce._cardinality):
                    new_triple = copy.deepcopy(self.prototype_edgetriple)
                    new_triple['start'] = str(dlsr.render(start_class))
                    new_triple['ids']['start'] = current_id
                    assert isinstance(
                        ce._property, OWLObjectProperty)
                    property_str = str(dlsr.render(ce._property))
                    new_triple['edge'] = property_str
                    self.current_position = 'end'
                    new_ids = copy.deepcopy(current_id)
                    print('new triple debug', new_triple)
                    self.list_current_edgetriples.append(new_triple)
                    find_edges(ce._filler, start_class, new_ids+1+i)
            elif isinstance(ce, OWLObjectIntersectionOf):
                for op in ce.operands():
                    find_edges(op, None, current_id)
            elif isinstance(ce, OWLObjectUnionOf):
                operands_list = list(ce._operands)
                subset_size = random.randint(1, len(operands_list))
                random_subset = random.sample(operands_list, subset_size)
                for op in random_subset:
                    find_edges(op, start_class, current_id)
            for triple in self.list_current_edgetriples:
                if not any(value is None for value in triple.values()):
                    self.list_edges.append(triple)
                    self.list_current_edgetriples.remove(triple)

        def renumber_edges():
            """
            input: triple + ids : {'start', 'end'}
            output: triple + new_id for PyG, counting only nodes of 1 type
            """
            count_nodetypes = {}
            new_triples = []
            for triple in self.list_edges:
                new_triple = copy.deepcopy(triple)
                if new_triple['start'] not in count_nodetypes:
                    count_nodetypes[new_triple['start']] = 0
                    start_id = 0
                else:
                    count_nodetypes[new_triple['start']] += 1
                    start_id = count_nodetypes[new_triple['start']]
                    start_id += 1
                if new_triple['end'] not in count_nodetypes:
                    count_nodetypes[new_triple['end']] = 0
                else:
                    count_nodetypes[new_triple['end']] += 1
                new_triple['ids']['start'] = start_id
                new_triple['ids']['end'] = count_nodetypes[new_triple['end']]
                new_triples.append(new_triple)
            return new_triples
        find_edges(new_ce)
        edges = renumber_edges()
        graphdict = {}
        print('debug edges', edges)
        print('debug edgelist', self.list_edges, self.list_current_edgetriples)
        for edge in edges:
            graphdict.update(self.add_edge_to_graphdict(edge, graphdict))
        for key in graphdict:
            list_ids_start, list_ids_end = graphdict[key]
            graphdict[key] = torch.tensor([list_ids_start, list_ids_end])
        print('debug gd', graphdict)
        graph = HeteroData()
        for key, values in graphdict.items():
            graph[key].edge_index = values
        print('debug', graph, graphdict)
        graph = GraphLibraryConverter.make_hdata_bidirected(graph)
        return graph


# --------------- old functions, possible stupid, maybe okay -------------------------------
def get_twisted_edge(edge):
    return (edge[2], edge[1], edge[0])


def are_dictionaries_identical(dict1, dict2):
    """
    Check if two dictionaries are identical (have the same keys and values).

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        bool: True if the dictionaries are identical, False otherwise.
    """
    # Check if the dictionaries have the same keys
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Check if the values for each key are the same
    for key in dict1:
        if dict1[key] != dict2[key]:
            return False

    # If all checks pass, the dictionaries are identical
    return True


def remove_front(s):
    if len(s) == 0:
        return s
    else:
        return s[len(xmlns):]


def make_graphdict_readable(old_dict):
    result = dict()
    for edge in old_dict.keys():
        if isinstance(edge[0], OWLClass):
            new_edge = (remove_front(edge[0].to_string_id()), remove_front(
                edge[1].to_string_id()), remove_front(edge[2].to_string_id()))
        else:
            new_edge = edge
        result[new_edge] = old_dict[edge]
    return result


def get_node_id_from_dict(graph_dict, node_type):
    """
    Gets the highest node id of a given node type from a graph dictionary.

    Parameters:
    graph_dict: The graph dictionary, in which the highest node id should be found.
    node_type: The node type, for which the highest node id should be found.

    Returns:
    max_id: The highest node id of the given node type.
    """
    try:
        class_name = remove_front(node_type.to_string_id())
    except:
        class_name = node_type
        pass
    max_id = -1
    for edge, ids in graph_dict.items():
        # TODO: Change the exceptions to checking directly on isinstance(edge, OWLClass)
        try:
            edge0 = remove_front(edge[0].to_string_id())
        except:
            edge0 = edge[0]
        try:
            edge2 = remove_front(edge[2].to_string_id())
        except:
            edge2 = edge[2]
        if edge0 == class_name:
            max_id = max(max_id, max(ids[0].tolist(), default=-1))
        if edge2 == class_name:
            max_id = max(max_id, max(ids[1].tolist(), default=-1))
    return max_id


def find_class(ce):
    new_class = None
    if isinstance(ce, OWLClass):
        return ce
    elif isinstance(ce, OWLObjectIntersectionOf):
        for op in ce.operands():
            new_class = find_class(op)
            if new_class is not None:
                break
        return new_class


def return_top_intersection(ce):
    if isinstance(ce, OWLObjectIntersectionOf):
        return ce
    elif isinstance(ce, OWLObjectUnionOf):
        for op in ce.operands():
            result = return_top_intersection(op)
            if result is not None:
                return result
    # TODO: Add OWLOBjectSomeValuesFrom


def length_ce(ce):
    # length_metric = OWLClassExpressionLengthMetric.get_default()
    return count_classes(ce)


def return_bottom_intersection(ce):
    if isinstance(ce, OWLObjectIntersectionOf):
        result = []
        for op in ce.operands():
            result.append(return_bottom_intersection(op))
        if all(value is None for value in result):
            return ce
        else:
            for op in ce.operands():
                result = return_bottom_intersection(op)
                if result is not None:
                    return result

        return ce
    elif isinstance(ce, OWLObjectUnionOf):
        for op in ce.operands():
            result = return_bottom_intersection(op)
            if result is not None:
                return result


def add_op_to_intersection(ce, new_op):
    list_of_operands = list(ce._operands)
    list_of_operands.append(new_op)
    ce._operands = tuple(list_of_operands)
    return ce


def add_op_to_intersection_deepcopy(ce: OWLObjectIntersectionOf, new_op):
    ce_new = copy.deepcopy(ce)
    list_of_operands = list(ce_new._operands)
    list_of_operands.append(new_op)
    ce_new._operands = tuple(list_of_operands)
    return ce_new


# function: Add sth to first intersection
def add_ce_to_top_intersect(ce, new_op):
    top_insec = return_top_intersection(ce)
    add_op_to_intersection(top_insec, new_op)
    return ce


def add_ce_to_bottom_intersect(ce, new_op):
    bottom_insec = return_bottom_intersection(ce)
    add_op_to_intersection(bottom_insec, new_op)
    return ce


def replace_property_of_fillers(ce):
    if isinstance(ce, OWLObjectIntersectionOf):
        for op in ce.operands():
            replace_property_of_fillers(op)
        return ce
    elif isinstance(ce, OWLObjectUnionOf):
        for op in ce.operands():
            replace_property_of_fillers(op)
        return ce
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        if isinstance(ce._property, list):
            ce._property = ce._property[0]
        return replace_property_of_fillers(ce._filler)
    return ce


def append_to_dict(dict, ids, edge):
    new_edge = (edge[0], edge[1], edge[2])
    twisted_edge = (edge[2], edge[1], edge[0])
    if new_edge not in dict:
        dict[new_edge] = (torch.tensor([ids[0]]), torch.tensor([ids[1]]))
        dict[twisted_edge] = (torch.tensor([ids[0]]), torch.tensor([ids[1]]))
        dict[new_edge] = (torch.tensor([ids[0]]), torch.tensor([ids[1]]))
        dict[twisted_edge] = (torch.tensor([ids[0]]), torch.tensor([ids[1]]))
    else:
        dict[new_edge] = (torch.cat((dict[new_edge][0], torch.tensor(ids[0]))),
                          torch.cat((dict[new_edge][1], torch.tensor(ids[1]))))
        dict[twisted_edge] = (torch.cat((dict[twisted_edge][0], torch.tensor(ids[1]))),
                              torch.cat((dict[twisted_edge][1], torch.tensor(ids[0]))))
        dict[new_edge] = (torch.cat((dict[new_edge][0], torch.tensor(ids[0]))),
                          torch.cat((dict[new_edge][1], torch.tensor(ids[1]))))
        dict[twisted_edge] = (torch.cat((dict[twisted_edge][0], torch.tensor(ids[1]))),
                              torch.cat((dict[twisted_edge][1], torch.tensor(ids[0]))))
    return dict


def update_dict(new_dict, old_dict):
    merged_dict = new_dict
    for edge, ids in old_dict.items():
        if edge not in merged_dict:
            merged_dict[edge] = ids
        else:
            merged_dict[edge] = (torch.cat((merged_dict[edge][0], ids[0])), torch.cat(
                (merged_dict[edge][1], ids[1])))
    return merged_dict


def is_valid_pyg_heterograph(graph_dict):
    """
    Checks whether a dictionary is a dictionary of a valid graph.
    A valid graph is a graph, in which every edge is represented twice, once in each direction.
    And the ids of the nodes are the same for the edge and its twisted edge.
    The edge is represented as a tuple of the form (startnode, edgetype, endnode).
    The ids are represented as a tuple of two lists. The first list contains the ids of the startnodes.
    The second list contains the ids of the endnodes.
    Example:
    graph_dict = {('A', 'edge', 'B'): (
        [0, 1, 2], [3, 4, 5]), ('B', 'edge', 'A'): ([0, 1, 2], [3, 4, 5])}
    This graph is not valid, because the ids of the startnodes and endnodes are not the same for the edge and its twisted edge.
    The correct representation would be:
    graph_dict = {('A', 'edge', 'B'): (
        [0, 1, 2], [3, 4, 5]), ('B', 'edge', 'A'): ([3, 4, 5], [0, 1, 2])}
    """
    # Check if the dictionary is a dictionary
    if not isinstance(graph_dict, dict):
        return False

    # Check if the dictionary is empty
    if len(graph_dict) == 0:
        return False
    is_valid = True
    for edge in graph_dict.keys():
        if get_twisted_edge(edge) not in graph_dict.keys():
            is_valid = False
            break
        elif edge[0] != edge[2]:
            if graph_dict[edge][0].tolist() != graph_dict[get_twisted_edge(edge)][1].tolist() or graph_dict[edge][1].tolist() != graph_dict[get_twisted_edge(edge)][0].tolist():
                is_valid = False
                break
        elif edge[0] == edge[2]:
            list1 = graph_dict[edge][0].tolist()
            list2 = graph_dict[edge][1].tolist()
            for a, b in zip(list1, list2):
                if (b, a) not in zip(list1, list2):
                    is_valid = False
                    break
        else:
            print('Case not implemented, please look into this.')
            pass
    return is_valid


def is_single_edge_graph(graph_dict):
    if len(graph_dict) != 1:
        return False
    if len(graph_dict[list(graph_dict.keys())[0]][0]) != 1:
        return False
    return True


def test_new_dict_on_useability(new_dict):
    # test, if all edges are used, as well as the twisted edges
    result_dict = {}
    for edge, ids in new_dict.items():
        result_dict[edge] = ids
        list_ids_0 = ids[0].tolist()
        list_ids_1 = ids[1].tolist()
        edge_start = edge[0]
        edge_end = edge[2]
        edge_middle = edge[1]
        if edge_start == edge_end:
            for a, b in zip(list_ids_0, list_ids_1):
                if (b, a) not in zip(list_ids_0, list_ids_1):
                    result_dict[edge] = (torch.cat((result_dict[edge][0], torch.tensor([b]))),
                                         torch.cat((result_dict[edge][1], torch.tensor([a]))))
        else:
            twisted_edge_found = False
            for edge2, ids2 in new_dict.items():
                if edge2[0] == edge_end and edge2[2] == edge_start and edge2[1] == edge_middle:
                    result_dict[edge2] = ids2
                    twisted_edge_found = True
                    if list_ids_0 == ids2[1].tolist() and list_ids_1 == ids2[0].tolist():
                        pass
                    else:
                        new_ids_end = torch.cat((ids[1], ids2[1]))
                        new_ids_start = torch.cat((ids[0], ids2[0]))
                        result_dict[edge] = (new_ids_start, new_ids_end)
                        result_dict[edge2] = (new_ids_end, new_ids_start)
            if twisted_edge_found is False:
                result_dict[(edge_end, edge_middle, edge_start)] = (
                    ids[1], ids[0])
    return result_dict


def update_current_ids(new_dict, nodetype, idstart, new_id):
    for edge2, ids2 in new_dict.items():
        if edge2[0] == nodetype:
            new_dict[edge2] = (torch.tensor(
                [new_id if id == idstart else id for id in ids2[0].tolist()]), ids2[1])
        if edge2[2] == nodetype:
            new_dict[edge2] = (torch.tensor(
                [new_id if id == idstart else id for id in ids2[0].tolist()]), ids2[1])
    return new_dict


def integrate_new_to_old_dict(old_dict, new_dict, current_node_id, update_id):
    """
    Integrates a new dictionary into an old dictionary. The old dictionary correctly represents a graph.
    The new dictionary is a graph, but it lacks correct node ids.
    And it may not be represented as a torch geometric heterodata object.

    Parameters:
    old_dict: The dictionary which already represents a graph correctly
    new_dict: The dictionary which represents a graph, but lacks correct node ids

    Returns:
    old_dict: The old dictionary, which now also contains the new dictionary
    current_node_id: The current node id of the last made entry. This is the highest node id of the old_dict
    """
    if not isinstance(old_dict, dict):
        raise TypeError("The argument old_dict must be a dictionary.")
    if not isinstance(new_dict, dict):
        raise TypeError("The argument new_dict must be a dictionary.")
    # check, if both dictionaries are not empty
    if len(old_dict) == 0 and len(new_dict) == 0:
        return new_dict, current_node_id
    elif len(new_dict) == 0:
        return old_dict, current_node_id

    is_valid_graph = is_valid_pyg_heterograph(new_dict)
    is_single_edge = is_single_edge_graph(new_dict)
    if is_valid_graph is True and is_single_edge is False:
        old_dict = update_dict(new_dict, old_dict)
        return old_dict, current_node_id
    # new edge has the startnode-id of its current_node_id and a new end-node-id
    # TODO: If only one edge is added, add this edge with current_node_id to new_node_id
    # TODO: Ensure, that this only created connected graphs
    # TODO: If a dictionary is merged to another dictionary, check, that the start-ids of the originally created edge stays the same and only the other node-id is adapted.
    for edge, ids in new_dict.items():
        list_ids_start = ids[0].tolist()
        for idstart in list_ids_start:
            idstart = int(idstart)
            # choose a random ID for the new node, but do not create an edge twice.
            new_id_max = get_node_id_from_dict(old_dict, node_type=edge[2])+1
            if edge[0] == edge[2]:
                new_id_max = max(new_id_max, max(ids[0].tolist())+1)
            if edge in old_dict:
                list_of_present_ids = old_dict[edge][1].tolist()
            else:
                list_of_present_ids = []
            list_range = list(range(int(new_id_max)+1))
            # TODO: Build in an option to regard this use and allow double-edges. (which have to be removed later)
            list_of_new_ids = [
                x for x in list_range if x not in set(list_of_present_ids)]
            if edge[0] == edge[2]:
                list_of_new_ids = [x for x in list_of_new_ids if x != idstart]
            new_id = random.choice(list_of_new_ids)
            new_dict_local = dict()
            # new_dict_local[edge] = (torch.Tensor([]), torch.Tensor([]))
            new_dict_local[edge] = torch.tensor([[idstart], [new_id]])
            # (torch.cat((torch.Tensor([idstart]), new_dict_local[edge][0])),
            # torch.cat((torch.Tensor([new_id]), new_dict_local[edge][1])))
            old_dict = update_dict(new_dict_local, old_dict)
    old_dict = make_graphdict_readable(old_dict)
    if update_id:
        return old_dict, new_id
    else:
        return old_dict, current_node_id

# -------------------------- generate ce dict from a ce


def create_graph_from_ce(ce, current_class, edgetype, current_node_id):
    """
    Converts a Class Expression (ce) into a graph dictionary as accurately as possible.

    Parameters:
    ce : Class Expression (could be OWLClass, OWLObjectIntersectionOf, etc.)
    current_class : Current OWL Class being processed
    edgetype : Current edge type between nodes
    current_node_id : ID of the current node being processed

    Returns:
    current_dict : A dictionary representing the graph (directed)
    current_node_id : Updated ID of the current node
    current_class : Updated OWL Class

    Assumptions:
    The ce is created in a way, that all individuals have a class.

    Attention:
    This only works for one edge-type currently, but not for multiple edge-types.
    """
    # Initialize an empty dictionary to hold the graph representation

    current_dict = dict()
    if isinstance(ce, OWLClass):
        current_class = ce
    elif isinstance(ce, OWLObjectIntersectionOf):
        if current_class is None:
            current_class = find_class(ce)
        if current_class is None:
            print('class to start with was not found, please look into this')
        else:
            for op in ce.operands():
                if not isinstance(op, OWLClass):
                    current_dict, current_node_id = integrate_new_to_old_dict(
                        current_dict,
                        create_graph_from_ce(
                            op, current_class, edgetype, current_node_id)[0],
                        current_node_id, update_id=False)
    elif isinstance(ce, OWLObjectSomeValuesFrom) and current_class is not None:
        # TODO: Sometimes, ce._property is a list, find out why and remove this case !!!
        if isinstance(ce._property, OWLObjectProperty):
            edgetype = ce._property
        # This is just for handling the error, that somewhere the property is saved as a list
        # instead of a ClassExpression
        if isinstance(ce._property, list):
            edgetype = ce._property[0]
        if isinstance(ce._filler, OWLClass):
            new_class = ce._filler
            new_node_id = 0
            new_dict = {(current_class, edgetype, new_class): (
                torch.tensor([current_node_id]), torch.tensor([new_node_id]))}
            new_dict = make_graphdict_readable(new_dict)
            current_dict, current_node_id = integrate_new_to_old_dict(
                current_dict, new_dict, current_node_id, update_id=True)
            current_class = new_class
        elif isinstance(ce._filler, OWLObjectIntersectionOf):
            new_class = find_class(ce._filler)
            new_node_id = 0
            new_dict = {(current_class, edgetype, new_class): (
                torch.tensor([current_node_id]), torch.tensor([new_node_id]))}
            new_dict = make_graphdict_readable(new_dict)
            current_dict, current_node_id = integrate_new_to_old_dict(
                current_dict, new_dict, current_node_id, update_id=True)
            current_class = new_class
            for op in ce._filler.operands():
                if not isinstance(op, OWLClass):
                    current_dict, current_node_id = integrate_new_to_old_dict(
                        current_dict,
                        create_graph_from_ce(
                            op, new_class, edgetype, current_node_id)[0],
                        current_node_id, update_id=False)
        else:
            print('Case not implemented yet')
            pass
    else:
        pass
    return current_dict, current_node_id, current_class


def get_graph_from_ce(ce, classtype, edgetype):
    """
    Attention:
    This only works for one edge-type currently, but not for multiple edge-types.
    """
    graph_dict = create_graph_from_ce(ce, classtype, edgetype, 0)[0]
    graph_dict = make_graphdict_readable(graph_dict)
    graph_dict = test_new_dict_on_useability(graph_dict)
    return graph_dict


def make_graphdict_from_pyg_hdata(hdata):
    metagraph = hdata.to_dict()
    graph_dict = dict()
    for edge, ids in metagraph.items():
        new_entry = list(ids.values())[0]
        new_entry = (new_entry[0], new_entry[1])
        graph_dict[edge] = new_entry
    keys_to_delete = [key for key in graph_dict.keys() if not (
        isinstance(key, tuple) and len(key) == 3)]
    for key in keys_to_delete:
        del graph_dict[key]
    return graph_dict


# Evalation Methods:

def find_adjacent(graph_dict, current_id, current_type):
    # graph_dict: dict of the form {[str,str,str] : [tensor,tensor]}, where the key represents (nodetype, edgetype, nodetype)-triples and the value the node-ids.
    # current_id: id of the current node
    # current_type: type of the current node
    # return: list of all adjacent nodes
    dict_adjacent = {}
    for key, ids in graph_dict.items():
        if key[0] == current_type and current_id in ids[0].tolist():
            existing_list = dict_adjacent.get(key[2], [])
            existing_list.extend(ids[1].tolist())
            existing_list = list(set(existing_list))
            dict_adjacent[key[2]] = existing_list
    return dict_adjacent


def fidelity_ce_recursive(ce, graph_dict, start_nodeid_graph, start_nodetype):
    # Startnodetype is always the class (in normal, not in OWL!)
    # The function is called with start_nodetype = None, by having done the check of it is the nodetype in the graph beforehand
    # Start: find top Class in CE, this should be adjacent to current startnodeid, startnodetype.
    #
    # Do the next code for every adjacent node found, until a true value is returned.
    # If no adjacent node is found, return false.
    # TODO Check, when to change the nodeids
    if isinstance(ce, OWLClass):
        class_for_graph = remove_front(ce.to_string_id())
        adjacent_nodes = find_adjacent(
            graph_dict, start_nodeid_graph, start_nodetype)
        if class_for_graph in adjacent_nodes.keys():
            return True
        else:
            return False
    elif isinstance(ce, OWLObjectIntersectionOf):
        dict_future_tests = {}
        next_class = find_class(ce)
        next_class_str = remove_front(next_class.to_string_id())
        if start_nodetype is None:
            start_nodetype = remove_front(next_class.to_string_id())
            adjacent_nodes = {start_nodetype: [start_nodeid_graph]}
            new_node_type = next_class
            for op in ce.operands():
                if not isinstance(op, OWLClass):
                    if not fidelity_ce_recursive(op, graph_dict, start_nodeid_graph, start_nodetype):
                        return False
            return True
        else:
            if fidelity_ce_recursive(next_class, graph_dict, start_nodeid_graph, start_nodetype):
                adjacent_nodes = find_adjacent(
                    graph_dict, start_nodeid_graph, start_nodetype)
                new_node_type = next_class
                for ind in adjacent_nodes[remove_front(new_node_type.to_string_id())]:
                    for op in ce.operands():
                        if not isinstance(op, OWLClass):
                            if not fidelity_ce_recursive(op, graph_dict, ind, new_node_type.to_string_id()):
                                return False

                return True
            else:
                return False

    elif isinstance(ce, OWLObjectSomeValuesFrom):
        # TODO: Check property to edgetype
        return fidelity_ce_recursive(ce._filler, graph_dict, start_nodeid_graph, start_nodetype)


def fidelity_ce_testdata(datasetfid, modelfid, ce_for_fid, node_type_expl, label_expl):
    # Start: Test for right start ID, then return a list of filler-ces
    fid_result = -1
    graph_dict = make_graphdict_from_pyg_hdata(datasetfid)
    mask = datasetfid[node_type_expl].test_mask
    mask_tf = 0
    for value in mask.tolist():
        if str(value) == 'True' or str(value) == 'False':
            mask_tf = 1
            break
    if label_expl == -1:
        list_labels = datasetfid[node_type_expl].y
        label_expl = max(set(list_labels))
    modelfid.eval()
    pred = modelfid(datasetfid.x_dict,
                    datasetfid.edge_index_dict).argmax(dim=-1)
    pred_list = pred.tolist()
    for index, value in enumerate(pred_list):
        if value != label_expl:
            pred_list[index] = 0
        else:
            pred_list[index] = 1
    pred = torch.tensor(pred_list)
    if mask_tf == 0:
        mask = datasetfid[node_type_expl]['test_mask']
        # cedict = generate_cedict_from_ce(ce_for_fid)
        smaller_mask = random.sample(
            mask.tolist(), k=min(200, len(mask.tolist())))
        mask = torch.tensor(smaller_mask)
    else:
        indices_of_ones = [i for i, value in enumerate(
            mask.tolist()) if value == True]
        chosen_indices = random.sample(
            indices_of_ones, k=min(20, len(indices_of_ones)))
        mask = [i if i in chosen_indices else 0 for i in range(
            len(mask.tolist()))]
        mask = [x for x in mask if x != 0]
        mask = torch.tensor(mask)
        sys.exit()
    count_fids = 0
    count_zeros_test = 0
    count_zeros_gnn = 0
    for index in mask.tolist():
        if isinstance(node_type_expl, OWLClass):
            node_type_expl = remove_front(node_type_expl.to_string_id())
        result_ce_fid = fidelity_ce_recursive(
            ce_for_fid, graph_dict, index, node_type_expl)
        # compute_prediction_ce(cedict, metagraph, node_type_expl, index)
        if pred[index] == result_ce_fid:
            count_fids += 1
        if result_ce_fid == 0:
            count_zeros_test += 1
        if pred[index] == 0:
            count_zeros_gnn += 1
    fid_result = round(float(count_fids) / float(len(mask.tolist())), 2)
    return fid_result
