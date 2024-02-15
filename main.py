# import GNN_playground #this is only locally important
# print('GNN_playground durchgelaufen')
import bashapes_model as bsm
from ManipulateScores import ManipulateScores
from datasets import create_hetero_ba_houses, initialize_dblp
from datasets import HeteroBAMotifDataset, GenerateRandomGraph, GraphMotifAugmenter
from generate_graphs import get_number_of_hdata, get_gnn_outs
from create_random_ce import random_ce_with_startnode, get_graph_from_ce, mutate_ce, length_ce, length_ce, fidelity_ce_testdata, replace_property_of_fillers
from visualization import visualize_hd
from evaluation import ce_score_fct, ce_confusion_iterative, fidelity_el, ce_fast_instance_checker, Accuracy_El
from models import HeteroGNNModel, HeteroGNNTrainer
from bashapes_model import HeteroGNNBA
from utils import delete_files_in_folder, remove_front, top_unique_scores, remove_integers_at_end, get_last_number
import torch
import statistics
# from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
import torch as th
import pickle
import os
from models import dblp_model
from torch_geometric.data import HeteroData
from random import randint
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import dgl
import colorsys
import random
import os
from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom, OWLObjectUnionOf
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import IRI
from owlapy.render import DLSyntaxObjectRenderer
import warnings
import sys
import copy
import pandas as pd
import re
import logging
import time
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


# -------- include values from shell script
try:
    end_length = sys.argv[1]
    number_of_ces = sys.argv[2]
    number_of_graphs = int(sys.argv[3])
    lambdaone = int(sys.argv[4])
    random_seed = int(sys.argv[5])

except Exception as e:
    print(f"Error deleting file: {e}")
    end_length = 5
    number_of_ces = 20
    number_of_graphs = 30
    lambdaone = 0.0  # controls the length of the CE
    random_seed = 1
# Further Parameters:
layers = 2  # 2 or 4 for the bashapes hetero dataset
start_length = 2

num_top_results = 10
save_ces = True  # Debugging: if True, creates new CEs and save the CEs to hard disk
# hyperparameters for scoring

lambdatwo = 0  # controls the variance in the CE output on the different graphs for one CE
aggr_fct = 'max'  # 'max' or 'mean'


# retraining Model and Dataset
# TODO: Create a list of pre-defined datasets with models
# For Code-Idea: Compare with GAIN-Presentation


# Which parts of the Code should be run?
retrain_GNN_and_dataset = True
run_beam_search = True
run_tests_ce = True


# tests
visualize_twice = True


# ----------------  utils


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
    list_results = []
    for _ in range(number_of_ces):
        local_dict_results = dict()
        ce_here = random_ce_with_startnode(length=start_length, typestart=OWLClass(
            IRI(NS, target_class)), list_of_classes=node_types, list_of_edge_types=edge_ces)
        assert isinstance(ce_here, OWLClassExpression), (ce_here, "This is not a Class Expression")
        local_dict_results['CE'] = ce_here
        list_graphs = get_number_of_hdata(ce_here, hd, num_graph_hdata=number_of_graphs)
        local_dict_results['GNN_outs'] = list()
        for graph in list_graphs:
            gnn_out = get_gnn_outs(graph, model, target_class)
            local_dict_results['GNN_outs'].append(gnn_out)
        score = ce_score_fct(ce_here, local_dict_results['GNN_outs'], lambdaone, lambdatwo)
        local_dict_results['score'] = score
        list_results.append(local_dict_results)
    list_sorted = sorted(list_results, key=lambda x: x['score'], reverse=True)
    list_results = list_sorted[:number_of_ces]
    for _ in range(start_length, end_length+1):
        # print(dlsr.render(replace_property_of_fillers(list_results[0]['CE'])))
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
            # end_time = time.time()
            # print(end_time-start_time)
            score = ce_score_fct(ce_here, local_dict_results['GNN_outs'], lambdaone, lambdatwo, aggregate=aggr_fct)
            local_dict_results['score'] = score
            new_list.append(local_dict_results)
        list_results = sorted(list_results + new_list, key=lambda x: x['score'], reverse=True)[:number_of_ces]
        print(f'Round {_} of {end_length} is finished')

    # only give as feedback the best different top results. Check this, by comparing different scores.
    # In any case, give as feedback 5 CEs, to not throw any errors in the code
    return list_results


def calc_fid_acc_top_results(list_results, model, target_class, dataset):
    if isinstance(target_class, OWLClassExpression):
        target_class = remove_front(target_class.to_string_id())
    for rdict in list_results:
        statistics_fidelity_dict = fidelity_el(ce=rdict['CE'], dataset=dataset,
                                               node_type_to_expl=target_class, model=model, label_to_expl=1)
        # fidelity = fidelity_ce_testdata(datasetfid=dataset, modelfid=model,
        #                                ce_for_fid=rdict['CE'], node_type_expl=target_class, label_expl=-1)

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
    three_to_two_to_one_to_one = OWLObjectIntersectionOf([class_3, edge_to_two_to_one_to_one])
    ce2 = three_to_two_to_one_to_one
    fidelity2 = fidelity_el(ce=ce2, dataset=dataset, node_type_to_expl=target_class, model=model, label_to_expl=1)
    print('Fidelity of the 3-2-1-1 CE is: ', fidelity2)
    ce_three_2 = OWLObjectIntersectionOf([class_3, edge_to_two])
    fidelity3 = fidelity_el(ce=ce_three_2, dataset=dataset,
                            node_type_to_expl=target_class, model=model, label_to_expl=1)
    print('Fidelity of 3-2: ', fidelity3)
    return list_results


def output_top_results(list_results, target_class, list_all_nt):
    # here we visualize with visualize_hd(hd_graph, addname_for_save, list_all_nodetypes, label_to_explain, add_info='', name_folder='')
    print('debug targer class: ', target_class)
    # begin debugging
    for ce_result in list_results:
        print(ce_result['score'])
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
        print('debugging: ', ce_result)
        for graph in ce_result['graphs']:
            str_addon_save = 'CE' + str(list_results.index(ce_result)) + '_Gr_' + str(ce_result['graphs'].index(graph))
            visualize_hd(graph, str_addon_save, list_all_nt, target_class, add_info=caption, name_folder='')


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
    list_results = calc_fid_acc_top_results(list_results, model, target_class, hd)
    output_top_results(list_results, target_class, hd.node_types)
    return list_results


# -----------------  Test new Datasets  -----------------------


def test_new_datasets(num_nodes=20000, num_motifs=2000, num_edges=3):
    # test the new datasets
    # create BA Graph
    ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(num_nodes=num_nodes, num_edges=num_edges)
    motif = {
        'labels': ['1', '2', '3', '4', '5'],
        'edges': [(0, 1), (0, 2), (0, 3), (0, 4)]
    }
    motif_house_letters = {
        'labels': ['A', 'B', 'B', 'C', 'C'],
        'edges': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    }
    type_to_classify = '2'
    synthetic_graph_class = GraphMotifAugmenter(motif='house', num_motifs=num_motifs, orig_graph=ba_graph_nx)
    synthetic_graph = synthetic_graph_class.graph
    # Workaround, fix later

    dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
    dataset_class.augmenter = synthetic_graph_class
    dataset = dataset_class._convert_labels_to_node_types()
    print(dataset)
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
    model = model_trainer.train(epochs=5000)
    # modelHeteroBSM = bsm.train_GNN(train_new_GNN, dataset, layers=layers)
    # modelHeteroBSM.eval()
    # end debug
    # not working still ....

    # start new debug: evaluate model on own dataset
    model.eval()
    return model


def dataset_debug():
    ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(num_nodes=500, num_edges=2)
    motif = 'house'
    type_to_classify = '3'
    synthetic_graph_class = GraphMotifAugmenter(motif=motif, num_motifs=2, orig_graph=ba_graph_nx)
    synthetic_graph = synthetic_graph_class.graph
    dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
    dataset = dataset_class._convert_labels_to_node_types()
    print(dataset)
    return dataset


def dataset_bashapes():
    bashapes = create_hetero_ba_houses(50, 10)
    dataset = bashapes
    dataset.num_node_types = 4
    dataset.type_to_classify = '3'
    dataset.type_to_classify = str(dataset.type_to_classify)  # to avoid errors
    return dataset


# test ce_creation
test_ce_new_datasets = True
if test_ce_new_datasets:
    if retrain_GNN_and_dataset:
        dataset, dataset_class = test_new_datasets()
        model = train_gnn_on_dataset(dataset)
        with open('content/dataset_bashapes.pkl', 'wb') as file:
            pickle.dump(dataset, file)
        with open('content/model_bashapes.pkl', 'wb') as file:
            pickle.dump(model, file)
    else:
        try:
            with open('content/dataset_bashapes.pkl', 'rb') as file:
                dataset = pickle.load(file)
            with open('content/model_bashapes.pkl', 'rb') as file:
                model = pickle.load(file)
        except Exception:
            print("Error loading dataset and model from file.")
            dataset, dataset_class = test_new_datasets()
            model = train_gnn_on_dataset(dataset)
            with open('content/dataset_bashapes.pkl', 'wb') as file:
                pickle.dump(dataset, file)
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
    run_tests_ce = True
    if run_tests_ce:
        with open('content/list_results.pkl', 'rb') as file:
            list_results = pickle.load(file)
        # change graphs and see if score changes.
        manipulated_scores = ManipulateScores(orig_scores=list_results, model=model,
                                              target_class=dataset.type_to_classify, aggregation=aggr_fct)
        manipulated_graphs = manipulated_scores.delete_list_of_graphs_1_3(list_results[0]['graphs'])
        new_scores = manipulated_scores.score_manipulated_graphs(list_results, manipulated_graphs)
        print('Running with manipulated scores: Done')
        print('We have used datasets of the following: ')
        print('Number of 3-1 edges: ', dataset['3', 'to', '1'].num_edges)
        print('Number of 3-1 edges, where 1 or 3 is in the motif: ', dataset_class.get_number_3_1_attached_to_motifs())

        # Further Test: Add 3-1 Edges
        run_test_add_3_1 = True
        if run_test_add_3_1:
            new_scores = manipulated_scores.score_add_3_1_edges(list_results)

        # Furthermore: Change graphs of best CEs and reevaluate the GNN Score.
        # This is done in the function: ce_confusion_iterative(ce, dataset, list_of_classes)

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
                assert isinstance(nt, str), (nt, "This is not a string, but should be")
                nt_str = str(nt)  # debugging: Ensure only strings are used
                node_types.append(OWLClass(IRI(NS, nt_str)))
            list_results = []
            for _ in range(number_of_ces):
                local_dict_results = dict()
                ce_here = ce_
                assert isinstance(ce_here, OWLClassExpression), (ce_here, "This is not a Class Expression")
                local_dict_results['CE'] = ce_here
                list_graphs = get_number_of_hdata(ce_here, hd, num_graph_hdata=number_of_graphs)
                local_dict_results['graphs'] = list_graphs
                local_dict_results['GNN_outs'] = list()
                for graph in list_graphs:
                    gnn_out = get_gnn_outs(graph, model, target_class)
                    local_dict_results['GNN_outs'].append(gnn_out)
                score = ce_score_fct(ce_here, local_dict_results['GNN_outs'], lambdaone, lambdatwo)
                local_dict_results['score'] = score
                list_results.append(local_dict_results)
                list_results = top_unique_scores(list_results, num_top_results)
                list_results = calc_fid_acc_top_results(list_results, model, target_class, hd)
                output_top_results(list_results, target_class, hd.node_types)
            print(list_results)


# ------------------  Testing Zone -----------------------
'''
ce_test_here = create_test_ce_3011()[1] #3-1-2
#print(919, dlsr.render(ce_test))
#ce_confusion(ce_test, motif = 'house') #should output 0
ce_last_class = find_last_class(ce_test_here)
print(ce_last_class)
print(1076, dlsr.render(ce_last_class))
ce_last_class = create_test_ce_3011()[1]
print(find_last_class(ce_test_here))
print(1076, dlsr.render(find_last_class(ce_test_here)))
print(1080, ce_to_tree_list(ce_test_here))

list_3011 = ['Intersection', ['3'], ['to', ['Intersection', ['0'], ['to', ['Intersection', ['1'], ['to', ['1']]]]]]]
'''
