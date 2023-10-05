import dgl
from torch_geometric.data import HeteroData
import torch
import torch_geometric
from create_random_ce import random_ce_with_startnode, get_graph_from_ce, mutate_ce
import random
from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import IRI
from owlapy.render import DLSyntaxObjectRenderer
import copy
# generete several graphdicts to one CE
# first, run multiple times the function for generating graphdicts
# if these don't yield different results, add a mutation to the CE and try again
dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns


def dict_in_list(new_dict, current_list):
    for existing_dict in current_list:
        # Compare dictionaries based on your specific structure.
        local_check = True
        for key, value in existing_dict.items():
            if key not in new_dict:
                local_check = False
                break
            if not torch.equal(new_dict[key][0], value[0]) or not torch.equal(new_dict[key][1], value[1]):
                local_check = False
                break
        if local_check:
            return True
    return False


def convert_tensors_to_int(input_dict):
    # Initialize an empty dictionary to store the results
    new_dict = {}
    # Iterate through the key-value pairs in the input dictionary
    for key, value in input_dict.items():
        # Convert the tensors to integers
        new_value = tuple(v.int() for v in value)
        # Add the converted key-value pair to the new dictionary
        new_dict[key] = new_value
    return new_dict


def iterate_to_find_graphdicts(current_list, ce, origdata, num_graphdicts, maximum_iterations=20):
    if len(current_list) >= num_graphdicts:
        return current_list
    else:
        for _ in range(maximum_iterations):
            new_dict = get_graph_from_ce(ce, None, [origdata.edge_types[0][1]])
            if not dict_in_list(new_dict, current_list):
                if len(current_list) < num_graphdicts:
                    current_list.append(new_dict)
        if len(current_list) >= num_graphdicts:
            return current_list
        else:
            if isinstance(origdata.node_types[0], str):
                list_ce_nodetypes = list()
                list_ce_edgetypes = list()
                for nodetype in origdata.node_types:
                    nt_ce = OWLClass(IRI(NS, nodetype))
                    list_ce_nodetypes.append(nt_ce)
                for edgetype in origdata.edge_types:
                    et_ce = OWLObjectProperty(IRI(NS, edgetype[1]))
                    list_ce_edgetypes.append(et_ce)
                list_ce_edgetypes = list(set(list_ce_edgetypes))
            else:
                list_ce_nodetypes = origdata.node_types
                list_ce_edgetypes = origdata.edge_types
            new_ce = mutate_ce(ce, list_ce_nodetypes, list_ce_edgetypes)
            return iterate_to_find_graphdicts(current_list, new_ce, origdata, num_graphdicts)


def generate_graphdicts_for_ce(ce, origdata, num_graphdicts=10):
    ce_new = copy.deepcopy(ce)
    graphdicts = iterate_to_find_graphdicts([], ce_new, origdata, num_graphdicts)
    return graphdicts


def heteroDatainfo(hetdata):
    list_n_types = hetdata.node_types
    node_types = []  # [[node_type, unique_values(int), #of features] for each nodetype]
    for nodet in list_n_types:  # create list [nodetype, unique values(int), size]
        list_features = torch.empty(2, 0)
        list_int_unique = list()
        try:
            list_features = hetdata[nodet].x
            list_int_unique = list(set([int(i) for i in list(list_features.unique())]))  # what does this do?
        except Exception as e:
            print(f"64 gg Here we skiped the error: {e}")
        node_types.append([nodet, list_int_unique, list_features.size(dim=1)])
    metapath_types = hetdata.edge_types  # saving possible meta-paths
    return node_types, metapath_types


def one_node_features(nodetype_list):
    rnd_feat_list = []
    for j in range(nodetype_list[2]):
        rnd_feat_list.append(int(random.choice(nodetype_list[1])))
    tensor_features = torch.tensor(rnd_feat_list, dtype=torch.float)
    return tensor_features


def all_node_features_one_type(nodename, dict_current_graph, hetdata):
    # call with the list for one nodetype, receive an appended random feature-vector
    node_info = heteroDatainfo(hetdata)[0]
    # obtain node_info_triplet for nodename
    node_info_triplet = []
    for triplets in node_info:
        if triplets[0] == nodename:
            node_info_triplet = triplets
            break
    # obtain number of nodes
    # TODO: Eigene Fkt schreiben, um dgl nicht zu verwenden
    num_nodes = dgl.heterograph(dict_current_graph).num_nodes(nodename)
    # create node_features
    list_node_features = []
    # for each node: call one_node_features
    # save the obtained tensors to a list
    for _ in range(num_nodes):
        list_node_features.append(one_node_features(node_info_triplet))
    # make a feature_matrix for this node_type
    feature_tensor_matrix = torch.stack(list_node_features)
    return feature_tensor_matrix


def create_features_to_dict(graph_dict, origdata):
    features_list = []
    # list of available nodetypes:
    graph_dict = convert_tensors_to_int(graph_dict)
    listntypes = dgl.heterograph(graph_dict).ntypes  # TODO: Durch eigene Funktion ersetzen, um nicht dgl zu verwenden
    for nodename in listntypes:
        features_list.append([nodename, all_node_features_one_type(nodename, graph_dict, origdata)])
    # features_list = [nodetype, features]
    return graph_dict, features_list


def graphdict_and_features_to_heterodata(graph_dict, features_list):
    hdata = HeteroData()
    # create features and nodes
    for name_tuple in features_list:
        name = name_tuple[0]
        hdata[name].x = name_tuple[1]
    # create edges
    # read from dict
    for edge in graph_dict:
        hdata[edge[0], edge[1], edge[2]].edge_index = torch.tensor([graph_dict[edge][0].tolist(),
                                                                    graph_dict[edge][1].tolist()], dtype=torch.long)
    return hdata


def get_number_of_hdata(ce, origdata, num_graph_hdata=10):
    graphdicts = generate_graphdicts_for_ce(ce, origdata, num_graph_hdata)
    hdata_list = []
    for graph_dict in graphdicts:
        hdata_list.append(graphdict_and_features_to_heterodata(
            graph_dict, create_features_to_dict(graph_dict, origdata)[1]))
    return hdata_list


def get_gnn_outs(hd, model, cat_to_explain):
    out = model(hd.x_dict, hd.edge_index_dict)
    if isinstance(cat_to_explain, str):
        cat_to_explain = -1
    elif isinstance(cat_to_explain, int):
        if cat_to_explain >= len(out[0]):
            cat_to_explain = -1
    elif isinstance(cat_to_explain, OWLClass):
        cat_to_explain = -1
    result = round(out[0][cat_to_explain].item(), 2)
    return result
