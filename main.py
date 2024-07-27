# import GNN_playground #this is only locally important
# print('GNN_playground durchgelaufen')
from datetime import datetime
import time
import logging
import re
import pandas as pd
import copy
import sys
import warnings
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import IRI
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom, OWLObjectUnionOf
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDataProperty
from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
import random
import colorsys
import dgl
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from random import randint
from torch_geometric.data import HeteroData
from models import dblp_model
import os
import pickle
import torch as th
import statistics
import torch
from evaluation import ce_score_fct, ce_confusion_iterative, ce_fidelity
from visualization import visualize_hd
from create_random_ce import random_ce_with_startnode, get_graph_from_ce, mutate_ce, length_ce, length_ce, fidelity_ce_testdata, replace_property_of_fillers
from generate_graphs import get_number_of_hdata, get_gnn_outs
from datasets import HeteroBAMotifDataset, GenerateRandomGraph, GraphMotifAugmenter
from datasets import create_hetero_ba_houses, initialize_dblp
from ManipulateScores from bashapes_model import HeteroGNNBA
import ManipulateScores
import bashapes_model as bsm
<< << << < HEAD
# from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset


# Generate a unique filename with the current date and time
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename_results_scoring = f'results_txt/scores_{timestamp}.txt'
with open(filename_results_scoring, 'w') as file:
    file.write('Scoring results for the CE: \n')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.WARNING)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*findfont.*")
dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns
random_seed = 1
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)  # For CUDA-enabled GPUs


# -------- include values from shell script
try:
    end_length = sys.argv[1]
    number_of_ces = sys.argv[2]
    number_of_graphs = int(sys.argv[3])
    lambdaone = int(sys.argv[4])
    lambdaone_fidelity = int(sys.argv[5])
    random_seed = int(sys.argv[6])

except Exception as e:
    print(f"Error deleting file: {e}")
<< << << < HEAD
end_length = 12
number_of_ces = 1000
number_of_graphs = 100
lambdaone = 1  # controls the length of the CE
lambdaone_fidelity = 0  # controls the length of the CE
random_seed = 1
# Further Parameters:
layers = 2  # 2 or 4 for the bashapes hetero dataset
size_dataset = 10000  # 10000 for the bashapes hetero dataset
start_length = 2
end_length = 10
number_of_ces = 1000
number_of_graphs = 10
num_top_results = 5
save_ces = True  # Debugging: if True, creates new CEs and save the CEs to hard disk
# hyperparameters for scoring
lambdaone = 0.2  # controls the length of the CE
lambdatwo = 0.0  # controls the variance in the CE output


# ----------------  utils


== == == =
run_DBLP = False
run_BAShapes = True
random_seed = 1
iterations = 3  # not implemented in newer version !!
# Further Parameters:
layers = 2  # 2 or 4 for the bashapes hetero dataset
start_length = 2
end_length = 10
number_of_ces = 1000
number_of_graphs = 10
num_top_results = 5
save_ces = True  # Debugging: if True, creates new CEs and save the CEs to hard disk
# hyperparameters for scoring

lambdatwo = 0  # controls the variance in the CE output on the different graphs for one CE
aggr_fct = 'max'  # 'max' or 'mean'


# retraining Model and Dataset
# TODO: Create a list of pre-defined datasets with models


# Which parts of the Code should be run?
retrain_GNN_and_dataset = False
run_beam_search = False
run_tests_ce = False
# Run on a new dataset with a new scoring function
run_beam_search_fidelity = True


# ----------------  utils


run_DBLP = False
run_BAShapes = True
random_seed = 1
iterations = 3  # not implemented in newer version !!
# Further Parameters:
train_new_GNN = True
layers = 2  # 2 or 4 for the bashapes hetero dataset
start_length = 2
end_length = 4
number_of_ces = 100
number_of_graphs = 1
num_top_results = 10
save_ces = True  # Debugging: if True, creates new CEs and save the CEs to hard disk
# hyperparameters for scoring

lambdatwo = 0  # controls the variance in the CE output on the different graphs for one CE
aggr_fct = 'mean'  # 'max' or 'mean'


# retraining Model and Dataset
# TODO: Create a list of pre-defined datasets with models
# For Code-Idea: Compare with GAIN-Presentation


# Which parts of the Code should be run?
retrain_GNN_and_dataset = False
run_beam_search = False
run_tests_ce = True


# ----------------  utils

def remove_front(s):
    if len(s) == 0:
        return s
    else:
        return s[len(xmlns):]


def uniquify(path, extension='.pdf'):
    if path.endswith("_"):
        path += '1'
        counter = 1
    while os.path.exists(path+extension):
        counter += 1
        while path and path[-1].isdigit():
            path = path[:-1]
        path += str(counter)
    return path


def dummy_fct():
    print('I am just a dummy function')


def remove_integers_at_end(string):
    pattern = r'\d+$'  # Matches one or more digits at the end of the string
    result = re.sub(pattern, '', string)
    return result


def get_last_number(string):
    pattern = r'\d+$'  # Matches one or more digits at the end of the string
    match = re.search(pattern, string)
    if match:
        last_number = match.group()
        return int(last_number)
    else:
        return None


def delete_files_in_folder(folder_path):
    """Deletes all files in the specified folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")


def graphdict_and_features_to_heterodata(graph_dict, features_list=[]):
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


def select_ones(numbers, num_ones_to_keep):
    ones_indices = [i for i, num in enumerate(numbers) if num == 1]

    if len(ones_indices) <= num_ones_to_keep:
        return numbers

    zeros_indices = random.sample(
        ones_indices, len(ones_indices) - num_ones_to_keep)

    for index in zeros_indices:
        numbers[index] = 0

    return numbers


def top_unique_scores(results, num_top_results):
    """
    Returns the top scores from all evaluated CEs, which are saved in the list results.
    Omits CEs with the same score.
    """
    # Sort the list of dictionaries by 'score' in descending order
    sorted_results = list(
        sorted(results, key=lambda x: x['score'], reverse=True))
    sorted_results = sorted_results[:num_top_results]
    return sorted_results


# ---------------  Find the best explanatory CEs for a dataset and GNN

# What does the function do: Here, we randomly create Class expressions and rank them, given their maximal GNN output. The scoring on the graphs is calculated but not taken into account Then we take these CEs, create graphs which fulfill them and evaluate the CEs and the graphs by fidelity, accuracy, ... . Then we visualize the created graphs.

# Parameters: dataset, GNN, how many graphs for each CE should be created, how many ces should be created (iterations)


def beam_search(hd, model, target_class, start_length, end_length, number_of_ces, number_of_graphs, top_results):
    """
    Performs beam search to generate a list of candidate class expressions (CEs) for a given target class.

    Args:
        hd (object): An object containing the node and edge types of a metagraph.
        model (object): A trained GNN model.
        target_class (str): The target class for which CEs are to be generated.
        start_length (int): The minimum length of the CEs to be generated.
        end_length (int): The maximum length of the CEs to be generated.
        number_of_ces (int): The number of CEs to be generated for each length.
        number_of_graphs (int): The number of graphs to be generated for each CE.
        top_results (int): The number of top-scoring CEs to be returned.

    Returns:
        list: A list of dictionaries containing the generated CEs, their GNN outputs, and their scores.
    """
    # retrieve global node and edge types  from metagraph

    node_types = []
    metagraph = hd.edge_types  # [str,str,str]
    edge_ces = []

    for mp in metagraph:
        edge_ces.append([OWLObjectProperty(IRI(NS, mp[1]))])
    for nt in hd.node_types:
        assert isinstance(nt, str), (nt, "This is not a string, but should be")
        nt_str = str(nt)  # debugging: Ensure only strings are used
        node_types.append(OWLClass(IRI(NS, nt_str)))
        node_types.append(OWLClass(IRI(NS, nt)))
    list_results = []
    for _ in range(number_of_ces):
        local_dict_results = dict()
        ce_here = random_ce_with_startnode(length=start_length, typestart=OWLClass(
            IRI(NS, target_class)), list_of_classes=node_types, list_of_edge_types=edge_ces)
        assert isinstance(ce_here, OWLClassExpression), (ce_here,
                                                         "This is not a Class Expression")
        assert isinstance(ce_here, OWLClassExpression), (ce_here,
                                                         "This is not a Class Expression")
        local_dict_results['CE'] = ce_here
        list_graphs = get_number_of_hdata(
            ce_here, hd, num_graph_hdata=number_of_graphs)
        local_dict_results['GNN_outs'] = list()
        for graph in list_graphs:
            gnn_out = get_gnn_outs(graph, model, target_class)
            print('Graph to Evaluate: ', graph)
            print('Features 3 of graph: ', graph[str(target_class)].x.dim())
            # print("Node type 0 tensor is None:", graph['0'].x is None)
            print("Node type 3 tensor is None:",
                  graph[str(target_class)].x is None)
            # print("Data type of node type 0 features:", type(graph['0'].x))
            print("Data type of node type 2 features:",
                  type(graph[str(target_class)].x))
            # print("Shape of features for node type 0:", graph['0'].x.shape)
            print("Shape of features for node type 2:",
                  graph[str(target_class)].x.shape)
            for edge_type, edge_index in graph.edge_index_dict.items():
                print(f"Edge type: {edge_type}")
                print(f"Edge index tensor: {edge_index}")
                print(f"Shape of edge index tensor: {edge_index.shape}")

            gnn_out = get_gnn_outs(graph, model, target_class)
            local_dict_results['GNN_outs'].append(gnn_out)
        score = ce_score_fct(
            ce_here, local_dict_results['GNN_outs'], lambdaone, lambdatwo)
        local_dict_results['score'] = score
        list_results.append(local_dict_results)
    list_sorted = sorted(list_results, key=lambda x: x['score'], reverse=True)
    list_results = list_sorted[:number_of_ces]
    for _ in range(start_length, end_length+1):
        new_list = list()
        for ce in list_results:
            ce_here = copy.deepcopy(ce['CE'])

            ce_here = mutate_ce(ce_here, node_types, edge_ces)
            local_dict_results = dict()
            local_dict_results['CE'] = ce_here
            list_graphs = get_number_of_hdata(
                ce_here, hd, num_graph_hdata=number_of_graphs)  # is some hdata object now
            local_dict_results['graphs'] = list_graphs
            assert 'graphs' in local_dict_results, "The key 'graphs' does not exist in ce_result"
            local_dict_results['GNN_outs'] = list()

            for graph in list_graphs:
                # generate output of GNN
                # generate the results for acc, f, ...
                gnn_out = get_gnn_outs(graph, model, target_class)
                local_dict_results['GNN_outs'].append(gnn_out)
            score = ce_score_fct(
                ce_here, local_dict_results['GNN_outs'], lambdaone, lambdatwo)
            local_dict_results['score'] = score
            new_list.append(local_dict_results)
        list_results = sorted(
            list_results + new_list, key=lambda x: x['score'], reverse=True)[:number_of_ces]
        print(f'Round {_} of {end_length} is finished')

    # only give as feedback the best different top results. Check this, by comparing different scores.
    # In any case, give as feedback 5 CEs, to not throw any errors in the code
    return list_results


def calc_fid_acc_top_results(list_results, model, target_class, dataset):
    if isinstance(target_class, OWLClassExpression):
        target_class = remove_front(target_class.to_string_id())
    for rdict in list_results:
        fidelity = fidelity_ce_testdata(datasetfid=dataset, modelfid=model,
                                        ce_for_fid=rdict['CE'], node_type_expl=target_class, label_expl=-1)

        # accuracy = ce_confusion_iterative(rdict['CE'], dataset, [target_class, 0])
        rdict['fidelity_dict'] = statistics_fidelity_dict
        rdict['fidelity'] = statistics_fidelity_dict['fidelity']
        accuracy_class_instance = Accuracy_El()
        accuracy = accuracy_class_instance.ce_accuracy_to_house(rdict['CE'])
        rdict['accuracy'] = round(accuracy, 3)
        # assert 'graphs' in rdict, "The key 'graphs' does not exist in ce_result"

    # test this function and delete later!!
    class_3 = OWLClass(IRI(NS, '3'))
    class_2 = OWLClass(IRI(NS, '2'))
    class_1 = OWLClass(IRI(NS, '1'))
    class_0 = OWLClass(IRI(NS, '0'))
    edge = OWLObjectProperty(IRI(NS, 'to'))
    edge_to_one = OWLObjectSomeValuesFrom(property=edge, filler=class_1)
    edge_to_two = OWLObjectSomeValuesFrom(property=edge, filler=class_2)
    edge_to_two_to_one = OWLObjectSomeValuesFrom(
        property=edge, filler=OWLObjectIntersectionOf([class_2, edge_to_one]))
    edge_to_two_to_one_to_one = OWLObjectSomeValuesFrom(
        property=edge, filler=OWLObjectIntersectionOf([class_2, edge_to_two_to_one]))
    three_to_two_to_one_to_one = OWLObjectIntersectionOf(
        [class_3, edge_to_two_to_one_to_one])
    ce2 = three_to_two_to_one_to_one
    fidelity2 = fidelity_el(
        ce=ce2, dataset=dataset, node_type_to_expl=target_class, model=model, label_to_expl=1)
    print('Fidelity of the 3-2-1-1 CE is: ', fidelity2)
    ce_three_2 = OWLObjectIntersectionOf([class_3, edge_to_two])
    fidelity3 = fidelity_el(ce=ce_three_2, dataset=dataset,
                            node_type_to_expl=target_class, model=model, label_to_expl=1)
    print('Fidelity of 3-2: ', fidelity3)
    return list_results


def output_top_results(list_results, target_class, list_all_nt):
    # here we visualize with visualize_hd(hd_graph, addname_for_save, list_all_nodetypes, label_to_explain, add_info='', name_folder='')
    # print('output top results: debug target class: ', target_class)
    # begin debugging
    # for ce_result in list_results:
    #    print(ce_result['score'])
    # end debugging
    for ce_result in list_results:
        assert 'graphs' in ce_result, "The key 'graphs' does not exist in ce_result"
        # first we create the fitting caption
        ce = ce_result['CE']
        ce = replace_property_of_fillers(ce)
        if not isinstance(ce, str):
            if hasattr(ce, 'to_string_id'):
                ce_str = ce.to_string_id()
            else:
                if isinstance(ce, OWLClassExpression):
                    ce_str = str(dlsr.render(ce))
                else:
                    print(type(ce))
        else:
            ce_str = ce
        caption = 'CE: ' + ce_str + '\n' + 'Score: ' + \
            str(ce_result['score']) + '\n' + 'Fidelity: ' + str(ce_result['fidelity']) + \
            '\n' + 'Accuracy: ' + str(ce_result['accuracy']) + \
            '\n' + 'recall: ' + str(ce_result['fidelity_dict']['recall']) + \
            'precision: ' + str(ce_result['fidelity_dict']['precision']) + \
            'tnr: ' + str(ce_result['fidelity_dict']['true_negative_rate'])
        # then, we create the fitting name for savings
        # in the form CE_x_Gr_y
        # where x is the number of the CE and y is the number of the graph
        for graph in ce_result['graphs']:
            str_addon_save = 'CE' + \
                str(list_results.index(ce_result)) + '_Gr_' + \
                str(ce_result['graphs'].index(graph))
            visualize_hd(graph, str_addon_save, list_all_nt,
                         target_class, add_info=caption, name_folder='')


def beam_and_calc_and_output(hd, model, target_class, start_length, end_length,
                             number_of_ces, number_of_graphs, num_top_results):
    """
    Performs beam search, calculates FID accuracy and outputs top results.

    Args:
        hd (object): Object containing the hyperparameters and data. (Heterodata object from PyG)
        model (object): Trained model to use for beam search.
        target_class (int): Target class for beam search.
        start_length (int): Starting length for beam search.
        end_length (int): Ending length for beam search.
        number_of_ces (int): Number of candidate entities to consider.
        number_of_graphs (int): Number of graphs to generate.
        top_results (int): Number of top results to output.
    Returns:
        None
    """
    if save_ces:  # Debugging: if True, creates new CEs and save the CEs to hard disk
        list_results = beam_search(hd, model, target_class, start_length, end_length,
                                   number_of_ces, number_of_graphs, num_top_results)
        # debugging:
    # For Debug: Save to hard disk and load again
        if not os.path.exists('content/Results'):
            os.makedirs('content/Results')
        with open('content/Results/my_list.pkl', 'wb') as f:
            pickle.dump(list_results, f)
    else:
        with open('content/Results/my_list.pkl', 'rb') as f:
            list_results = pickle.load(f)
    # Only return the top num_top_results which are pairwise different, measured by the score
    list_results = top_unique_scores(list_results, num_top_results)
    list_results = calc_fid_acc_top_results(
        list_results, model, target_class, hd)
    output_top_results(list_results, target_class, hd.node_types)
    return list_results


# -----------------  Test new Datasets and then delete this part -----------------------


def test_new_datasets(num_nodes=10000, num_motifs=2000):
    # test the new datasets
    # create BA Graph
    ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(
        num_nodes=num_nodes, num_edges=3)
    motif = {
        'labels': ['1', '2', '3', '4', '5'],
        'edges': [(0, 1), (0, 2), (0, 3), (0, 4)]
    }
    motif_house_letters = {
        'labels': ['A', 'B', 'B', 'C', 'C'],
        'edges': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    }
    type_to_classify = '3'
    synthetic_graph_class = GraphMotifAugmenter(
        motif='house', num_motifs=num_motifs, orig_graph=ba_graph_nx)
    synthetic_graph = synthetic_graph_class.graph
    # Workaround, fix later

    dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
    dataset_class.augmenter = synthetic_graph_class
    dataset = dataset_class._convert_labels_to_node_types()
    print(dataset)
    return dataset


def train_gnn_on_dataset(dataset):
    # train GNN on dataset
    """
    # example test
    bashapes = create_hetero_ba_houses(500, 100)
    dataset = bashapes
    dataset.num_node_types = 4
    dataset.type_to_classify = '3'
    dataset.type_to_classify = str(dataset.type_to_classify)  # to avoid errors
    # end example test
    """

    model = HeteroGNNModel(dataset.metadata(), hidden_channels=16, out_channels=dataset.num_node_types,
                           node_type=dataset.type_to_classify, num_layers=2)
    model_trainer = HeteroGNNTrainer(model, dataset, learning_rate=0.01)
    model_trainer.train(epochs=20)
    return model


dataset = test_new_datasets()
model = train_gnn_on_dataset(dataset)


# test ce_creation
# beam_and_calc_and_output(hd=dataset, model=model, target_class='2',
#                         start_length=start_length, end_length=end_length,
#                         number_of_ces=number_of_ces, number_of_graphs=number_of_graphs,
#                         num_top_results=num_top_results)


# -----------------  Test new Datasets  -----------------------


def test_new_datasets(num_nodes=size_dataset, num_motifs=int(0.1*size_dataset), num_edges=3):
    # test the new datasets
    # create BA Graph
    ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(
        num_nodes=num_nodes, num_edges=num_edges)
    motif = {
        'labels': ['1', '2', '3', '4', '5'],
        'edges': [(0, 1), (0, 2), (0, 3), (0, 4)]
    }
    motif_house_letters = {
        'labels': ['A', 'B', 'B', 'C', 'C'],
        'edges': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    }
    type_to_classify = '2'
    synthetic_graph_class = GraphMotifAugmenter(
        motif='house', num_motifs=num_motifs, orig_graph=ba_graph_nx)
    synthetic_graph = synthetic_graph_class.graph
    # Workaround, fix later

    dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
    dataset_class.augmenter = synthetic_graph_class
    dataset = dataset_class._convert_labels_to_node_types()
    return dataset, dataset_class


def train_gnn_on_dataset(dataset):
    # ensure: out_channels = 2
    # train GNN on dataset
    """
    # example test
    bashapes = create_hetero_ba_houses(500, 100)
    dataset = bashapes
    dataset.num_node_types = 4
    dataset.type_to_classify = '3'
    dataset.type_to_classify = str(dataset.type_to_classify)  # to avoid errors
    # end example test
    """

    model = HeteroGNNModel(dataset.metadata(), hidden_channels=64, out_channels=2,
                           node_type=dataset.type_to_classify, num_layers=2)
    model_trainer = HeteroGNNTrainer(model, dataset, learning_rate=0.01)
    model = model_trainer.train(epochs=1000)
    # modelHeteroBSM = bsm.train_GNN(train_new_GNN, dataset, layers=layers)
    # modelHeteroBSM.eval()
    # end debug
    # not working still ....

    # start new debug: evaluate model on own dataset
    model.eval()
    return model


def dataset_debug():
    ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(
        num_nodes=500, num_edges=2)
    motif = 'house'
    type_to_classify = '3'
    synthetic_graph_class = GraphMotifAugmenter(
        motif=motif, num_motifs=2, orig_graph=ba_graph_nx)
    synthetic_graph = synthetic_graph_class.graph
    dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
    dataset = dataset_class._convert_labels_to_node_types()
    return dataset


def dataset_bashapes():
    bashapes = create_hetero_ba_houses(50, 10)
    dataset = bashapes
    dataset.num_node_types = 4
    dataset.type_to_classify = '3'
    dataset.type_to_classify = str(dataset.type_to_classify)  # to avoid errors
    return dataset


# test ce_creation
number_to_letter = {
    0: 'D',
    1: 'C',
    2: 'B',
    3: 'A'
}

test_ce_new_datasets = True
if test_ce_new_datasets:
    if retrain_GNN_and_dataset:
        dataset, dataset_class = test_new_datasets()
        model = train_gnn_on_dataset(dataset)
        with open('content/dataset_bashapes.pkl', 'wb') as file:
            pickle.dump(dataset, file)
        with open('content/dataset_class_bashapes.pkl', 'wb') as file:
            pickle.dump(dataset_class, file)
        with open('content/model_bashapes.pkl', 'wb') as file:
            pickle.dump(model, file)
    else:
        try:
            with open('content/dataset_bashapes.pkl', 'rb') as file:
                dataset = pickle.load(file)
            with open('content/dataset_class_bashapes.pkl', 'rb') as file:
                dataset_class = pickle.load(file)
            with open('content/model_bashapes.pkl', 'rb') as file:
                model = pickle.load(file)
        except Exception:
            print("Error loading dataset and model from file.")
            dataset, dataset_class = test_new_datasets()
            model = train_gnn_on_dataset(dataset)
            with open('content/dataset_bashapes.pkl', 'wb') as file:
                pickle.dump(dataset, file)
            with open('content/dataset_class_bashapes.pkl', 'wb') as file:
                pickle.dump(dataset_class, file)
            with open('content/model_bashapes.pkl', 'wb') as file:
                pickle.dump(model, file)
    out = model(dataset.x_dict, dataset.edge_index_dict)

    test_gt = True
    if run_beam_search:
        delete_files_in_folder('content/plots')
        list_results = beam_and_calc_and_output(hd=dataset, model=model, target_class=dataset.type_to_classify,
                                                start_length=2, end_length=end_length,
                                                number_of_ces=number_of_ces, number_of_graphs=number_of_graphs,
                                                num_top_results=num_top_results)
        HeteroBAMotifDataset.print_statistics_to_dataset(dataset)
        with open('content/list_results.pkl', 'wb') as file:
            pickle.dump(list_results, file)

    if run_tests_ce:
        with open('content/list_results.pkl', 'rb') as file:
            list_results = pickle.load(file)
        # Save to some file
        # with open(filename_results_scoring, 'a') as file:
        #    file.write('Showing changed scores for the CE: ' + str(list_results[0]['CE']) + '\n')
        #    file.write('Showing, if Edges A-C are deleted' + '\n')
        # change graphs and see if score changes.
        manipulated_scores = ManipulateScores(orig_scores=list_results, model=model,
                                              target_class=dataset.type_to_classify, aggregation=aggr_fct, file_name=filename_results_scoring)
        manipulated_graphs = manipulated_scores.delete_list_of_graphs_1_3(
            list_results[0]['graphs'])
        new_scores = manipulated_scores.score_manipulated_graphs(
            manipulated_graphs)
        print('Running with manipulated scores: Done')
        print('We have used datasets of the following: ')
        print('Number of 3-1 edges: ', dataset['3', 'to', '1'].num_edges)
        print('Number of 3-1 edges, where 1 or 3 is in the motif: ',
              dataset_class.get_number_3_1_attached_to_motifs())

        # Further Test: Add 3-1 Edges
        run_test_add_3_1 = False
        if run_test_add_3_1:
            manipulated_graphs = manipulated_scores.list_of_graphs_add_3_1_edges(
                list_results[0]['graphs'])
            # with open(filename_results_scoring, 'a') as file:
            #    file.write('Showing, if Edges A-C are deleted' + '\n')
            new_scores = manipulated_scores.score_manipulated_graphs(
                manipulated_graphs)

        # Furthermore: Change graphs of best CEs and reevaluate the GNN Score.
        # This is done in the function: ce_confusion_iterative(ce, dataset, list_of_classes)

        # run all in one loop, specific for the house dataset
        for ith_score in range(0, min(num_top_results, len(list_results))):
            print(f'Scoring the {ith_score}th best CE')
            with open(filename_results_scoring, 'a') as file:
                file.write(f'Scoring the {ith_score}th best CE' + '\n')
            for i in range(0, 4):
                for j in range(i, 4):
                    with open(filename_results_scoring, 'a') as file:
                        # print('debug: ', len(list_results), ith_score)
                        file.write('\tShowing changed scores for the CE: ' +
                                   str(dlsr.render(list_results[ith_score]['CE'])) + '\n')
                        file.write(
                            f'\t\tShowing, if edges {number_to_letter[i]}-{number_to_letter[j]} are deleted' + '\n')
                    manipulated_graphs = manipulated_scores.delete_list_of_graphs_i_j(
                        list_graphs=list_results[ith_score]['graphs'], i=i, j=j)
                    new_scores = manipulated_scores.score_manipulated_graphs(
                        manipulated_graphs)
                    with open(filename_results_scoring, 'a') as file:
                        file.write('\tShowing changed scores for the CE: ' +
                                   str(dlsr.render(list_results[ith_score]['CE'])) + '\n')
                        file.write(
                            f'\t\tShowing, if edges {number_to_letter[i]}-{number_to_letter[j]} are added' + '\n')
                    manipulated_graphs = manipulated_scores.list_of_graphs_add_i_j_edges(graphs_to_change=list_results[ith_score]['graphs'],
                                                                                         i=i, j=j)
                    new_scores = manipulated_scores.score_manipulated_graphs(
                        manipulated_graphs)

    run_tests_ce = False
    if run_tests_ce:
        # create CE
        class_3 = OWLClass(IRI(NS, '3'))
        class_2 = OWLClass(IRI(NS, '2'))
        class_1 = OWLClass(IRI(NS, '1'))
        class_0 = OWLClass(IRI(NS, '0'))
        edge = OWLObjectProperty(IRI(NS, 'to'))
        edge_to_one = OWLObjectSomeValuesFrom(property=edge, filler=class_1)
        edge_to_two = OWLObjectSomeValuesFrom(property=edge, filler=class_2)
        edge_to_two_to_one = OWLObjectSomeValuesFrom(
            property=edge, filler=OWLObjectIntersectionOf([class_2, edge_to_one]))
        # 3 - (intersection(2, -2,-1))
        gt_ce = OWLObjectIntersectionOf(
            [class_3,
             OWLObjectSomeValuesFrom(property=edge, filler=OWLObjectIntersectionOf([class_2, edge_to_one, edge_to_two])), edge_to_two_to_one])
        print('Here we print the Ce')
        print(dlsr.render(gt_ce))
        # score for gt CE:
        ce_3_2 = OWLObjectIntersectionOf([class_3, edge_to_two])
        ce_3 = class_3
        for ce_ in [ce_3_2, gt_ce]:
            hd = dataset
            node_types = []
            metagraph = hd.edge_types  # [str,str,str]
            edge_ces = []
            target_class = '3'
            number_of_ces = 1
            number_of_graphs = 10
            num_top_results = 1

            for mp in metagraph:
                edge_ces.append([OWLObjectProperty(IRI(NS, mp[1]))])
            for nt in hd.node_types:
                assert isinstance(
                    nt, str), (nt, "This is not a string, but should be")
                nt_str = str(nt)  # debugging: Ensure only strings are used
                node_types.append(OWLClass(IRI(NS, nt_str)))
            list_results = []
            for _ in range(number_of_ces):
                local_dict_results = dict()
                ce_here = ce_
                assert isinstance(
                    ce_here, OWLClassExpression), (ce_here, "This is not a Class Expression")
                local_dict_results['CE'] = ce_here
                list_graphs = get_number_of_hdata(
                    ce_here, hd, num_graph_hdata=number_of_graphs)
                local_dict_results['graphs'] = list_graphs
                local_dict_results['GNN_outs'] = list()
                for graph in list_graphs:
                    gnn_out = get_gnn_outs(graph, model, target_class)
                    local_dict_results['GNN_outs'].append(gnn_out)
                score = ce_score_fct(
                    ce_here, local_dict_results['GNN_outs'], lambdaone, lambdatwo)
                local_dict_results['score'] = score
                list_results.append(local_dict_results)
                list_results = top_unique_scores(list_results, num_top_results)
                list_results = calc_fid_acc_top_results(
                    list_results, model, target_class, hd)
                output_top_results(list_results, target_class, hd.node_types)
            print(list_results)


# ---------------------- evaluation DBLP
# this does not work accordingly

if run_DBLP:
    delete_files_in_folder('content/plots/DBLP')
    datadblp, targetdblp = initialize_dblp()
    retrain = False  # set to True, if Dataset should be retrained.
    modeldblp = dblp_model(retrain)
    # create_ce_and_graphs_to_heterodataset(datadblp, modeldblp, 'DBLP', targetdblp, cat_to_explain=-1,
    #                                     graphs_per_ce=16, iterations=iterations, compute_acc=False)


# ---------------------- evaluation HeteroBAShapes
# run_BAShapes = True
if run_BAShapes:
    # TODO: Save BAShapes to some file, code already somewhere
    bashapes = create_hetero_ba_houses(500, 100)
    delete_files_in_folder('content/plots')
    # train GNN 2_hop
    modelHeteroBSM = bsm.train_GNN(False, bashapes, layers=layers)
    start_time = time.time()
    beam_and_calc_and_output(hd=bashapes, model=modelHeteroBSM, target_class='3',
                             start_length=start_length, end_length=end_length,
                             number_of_ces=number_of_ces, number_of_graphs=number_of_graphs,
                             num_top_results=num_top_results)
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Parameters:\n"
          f"  Layers: {layers}\n"
          f"  Start Length: {start_length}\n"
          f"  End Length: {end_length}\n"
          f"  Number of CEs: {number_of_ces}\n"
          f"  Number of Graphs: {number_of_graphs}\n"
          f"  Num Top Results: {num_top_results}\n"
          f"  Save CEs: {save_ces}\n"
          f"  Lambda One: {lambdaone}\n"
          f"  Lambda Two: {lambdatwo}\n"
          f"Runtime: {elapsed_time:.4f} seconds")
    # create_ce_and_graphs_to_heterodataset(bashapes, modelHeteroBSM, 'HeteroBAShapes_BA2hop', '3', cat_to_explain=-1,
    #                                      graphs_per_ce=16, iterations=iterations, compute_acc=True, random_seed=random_seed)
    # train GNN 4_hop: Later, we can use this to compare the results of the 2_hop and 4_hop GNNs
    # modelHeteroBSM = bsm.train_GNN(True, bashapes, layers=4)
    # create_ce_and_graphs_to_heterodataset(bashapes, modelHeteroBSM, 'HeteroBAShapes_BA4hop', '3', cat_to_explain=-1,
    #                                      graphs_per_ce=16, iterations=iterations, compute_acc=True, random_seed=random_seed)


# ------------ Testing zone for CEs
