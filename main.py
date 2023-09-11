#import GNN_playground #this is only locally important
#print('GNN_playground durchgelaufen')
import bashapes_model as bsm
from datasets import create_hetero_ba_houses, initialize_dblp 
from graph_generation import create_graphs_for_heterodata, add_features_and_predict_outcome, compute_accu, compute_prediction_ce, compute_confusion_for_ce_line, compute_f1
from ce_generation import create_graphdict_from_ce, length_ce, create_test_ce_3011, create_test_ce_3012, generate_cedict_from_ce
from ce_generation import create_random_ce_from_BAHetero, remove_front, create_random_ce
from visualization import visualize_best_ces
from evaluation import ce_score_fct, ce_confusion_iterative, ce_fidelity
import torch
import statistics
# from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
import torch as th
from models import dblp_model
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, DBLP
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, to_hetero
import torch_geometric
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

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*findfont.*")
dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns
#random_seed = 3006
#random.seed(random_seed)



# -------- include values from shell script
try:
    run_DBLP = sys.argv[1]
    run_BAShapes = sys.argv[2]
    random_seed = int(sys.argv[3])
    iterations = int(sys.argv[4])
    if run_DBLP == "True":
        run_DBLP = True
    else:
        run_DBLP = False
    if run_BAShapes == "True":
        run_BAShapes = True
    else:
        run_BAShapes = False
except Exception as e:
    print(f"Error deleting file: {e}")    
    run_DBLP = True
    run_BAShapes = True
    random_seed = 1
    iterations = 3





# ----------------  utils 


def uniquify(path, extension = '.pdf'):
    if path.endswith("_"):
        path += '1'
        counter = 1
    while os.path.exists(path+extension):
        counter +=1
        while path and path[-1].isdigit():
            path = path[:-1]
        path +=str(counter)
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

            
def graphdict_and_features_to_heterodata(graph_dict, features_list = []):
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
    
    zeros_indices = random.sample(ones_indices, len(ones_indices) - num_ones_to_keep)
    
    for index in zeros_indices:
        numbers[index] = 0
    
    return numbers

def available_edges_with_nodeid(graph, current_type, current_id, edgetype = 'to'):
    # graph is in dictionary form
    list_result = list()
    for key, value in graph.items():
        if key[0] == current_type and key[1] == edgetype:
            for _, indexvalue in enumerate(value[0].tolist()):
                if current_id == indexvalue:
                    list_result.append([key[2], value[1].tolist()[_], value[2][_]])
    return list_result




        

# ---------------  Find the best explanatory CEs for a dataset and GNN
            
# What does the function do: Here, we randomly create Class expressions and rank them, given their maximal GNN output. The scoring on the graphs is calculated but not taken into account Then we take these CEs, create graphs which fulfill them and evaluate the CEs and the graphs by fidelity, accuracy, ... . Then we visualize the created graphs.
#TODO: Visualize only different graphs.
# TODO: Score the graphs not by GNN-output, but by our pre-defined scoring function.
# Parameters: dataset, GNN, how many graphs for each CE should be created, how many ces should be created (iterations)
def create_ce_and_graphs_to_heterodataset(hd, model, dataset_name, target_class, cat_to_explain = -1, graphs_per_ce = 10, iterations = 30, compute_acc = False, random_seed = 806):
    saved_graphs_ces = "content/created_graphs/"+'CEs_for_' + dataset_name
    #node types
    node_types = hd.node_types
    metagraph = hd.edge_types #[str,str,str]
    edge_names = []
    list_classes_objprops = []
    for mp in metagraph:
        if mp[1] not in edge_names:
            edge_names.append(mp[1])
            list_classes_objprops.append([OWLClass(IRI(NS, mp[0])), [OWLObjectProperty(IRI(NS, mp[1]))]])
            list_classes_objprops.append([OWLClass(IRI(NS, mp[2])), [OWLObjectProperty(IRI(NS, mp[1]))]])
    list_length_of_ce = [3,4,5,6,7,8]
    result_graphs = list()
    result_graphs_acc = list()
    result_ces = list()
    for _ in range(iterations):
        #try:
        random_seed +=9900
        random.seed(random_seed)
        len_ce = random.choice(list_length_of_ce)
        #ce_metagraph = create_metagraph_for_ce(metagraph)
        #print(884, ce_metagraph, target_class)
        ce_here = create_random_ce(list_classes_objprops, OWLClass(IRI(NS, target_class)), len_ce, random_seed)
        #ce_here = create_random_ce_from_metagraph(len_ce, metagraph, target_class, random_seed)
        local_dict_results = dict()
        local_dict_results['acc'] = []
        local_dict_results['f'] = []
        local_dict_results['GNN_outs'] = []
        for griter in range(graphs_per_ce):
            dict_graph = create_graphdict_from_ce(ce_here, node_types, edge_names, metagraph, random_seed) #possible nodes, edges have to be transmitted; features can be completed afterwards
            #on all: Calculate gnn and save the outs
            ce_str = ce_here
            try:
                ce_str = dlsr.render(ce_here)
            except Exception as e:
                print(f"789 Here we skiped the error: {e}")
            list_results = list()
            result = add_features_and_predict_outcome(1 ,cat_to_explain,  model, hd, list_results, dict_graph, saved_graphs_ces, ce_str = ce_str, compute_acc=compute_acc, random_seed=random_seed)
            local_dict_results['GNN_outs'].append(result[0][1])
            # calculate accuracy
            local_dict_results['acc'].append(result[0][3])
            local_dict_results['f'].append(result[0][3]) 
        # we need a result: [CE, averaged graph accuracy, CE_score, 1-3 example graphs]
        list_gnn_outs = local_dict_results['GNN_outs']
        lambdaone = 0.04
        lambdatwo = 0.08
        score = ce_score_fct(ce_here, list_gnn_outs, lambdaone, lambdatwo)
        list_accs =  [inner_list[0] for inner_list in local_dict_results['acc']]
        avg_graph_acc = sum(list_accs) / len(list_accs)
        max_graph_acc = max(list_accs)
        list_f =  [inner_list[2] for inner_list in local_dict_results['f']]
        fscore = max(list_f)
        #print(530, list_accs, avg_graph_acc, len(list_accs), sum(list_accs))
        #idea: create 10 example graphs and save top 3 graphs here
        local_result_ce = [ce_here, score, avg_graph_acc, max_graph_acc, fscore]
        print(883, dlsr.render(ce_here))
        add_ce = True
        for ce_prev in result_ces:
            if ce_prev[0] == ce_here:
                add_ce = False
        if add_ce == True:
            result_ces.append(local_result_ce) 
    result_ces = sorted(result_ces, key=lambda x: x[1], reverse = True)
    local_graphs_for_ce = list()
    random_seed = 7
    random.seed(random_seed)
    total_graph_to_ce_list = list()
    #TODO: Only range this for-loop for the number of graphs visualized. This can only be done, if hte CEs above are already ranked by score!
    for ceindex in range(len(result_ces)):
        local_graphs_for_ce = list()
        cel_best = result_ces[ceindex][0]
        ce_score = round(result_ces[ceindex][1],2)
        ce_avg_graph = round(result_ces[ceindex][2],2)
        ce_max_graph = round(result_ces[ceindex][3],2)
        ce_f = round(result_ces[ceindex][4],2)
        cel_str = cel_best
        try: 
            cel_str = dlsr.render(cel_best)
        except Exception as e:
            print(f"870 Here we skiped the error: {e}")
        dict_graph = dict()
        new_graph = list()
        for _ in range(5):
            random_seed +=1700
            random.seed(random_seed)
            dict_graph = create_graphdict_from_ce(cel_best, node_types, edge_names, metagraph, random_seed)
            random_seed +=1
            random.seed(random_seed)
            new_graph = add_features_and_predict_outcome(1 ,cat_to_explain,  model, hd, list(), dict_graph, saved_graphs_ces, ce_str = cel_str,compute_acc=compute_acc, random_seed = random_seed) #Here: TODO: give as feedback different graphs this time, not only the best graphs.
            local_graphs_for_ce += new_graph
        local_graphs_for_ce = sorted(local_graphs_for_ce, key=lambda x: x[1], reverse = True)
        total_graph_to_ce_list.append([local_graphs_for_ce, cel_str,ce_score, ce_avg_graph, ce_max_graph, ce_f,cel_best])
    print(576, cel_str,ce_score, ce_avg_graph, ce_max_graph)
    total_graph_to_ce_list = sorted(total_graph_to_ce_list, key=lambda x: x[2], reverse = True)
    visualize_best_ces(num_top_ces = 6, num_top_graphs = 3, list_of_results_ce = total_graph_to_ce_list, list_all_nodetypes = node_types, label_to_explain = target_class, ml=model, ds=hd, node_expl = target_class, plotname = dataset_name, random_seed = random_seed)
    
    
    
# ------------------  Testing Zone -----------------------    
    
def test_ce_prediction():
    ce_test = create_test_ce()
    graph_test =  {('3', 'to', '2') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long)),
                     ('2', 'to', '3') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long)),
                ('2', 'to', '1') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long)),
                     ('1', 'to', '2') :(torch.tensor([0], dtype = torch.long)),
                     }
    print(compute_prediction_ce(ce_dict=ce_test, graph_dict=graph_test, node_type_to_explain='3', index_to_explain=0))
    
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




    
    
    

#---------------------- evaluation DBLP 
# this does not work accordingly
random_seed = 3006
if run_DBLP:
    delete_files_in_folder('content/plots/DBLP')
    datadblp, targetdblp = initialize_dblp()
    retrain = False  # set to True, if Dataset should be retrained.
    modeldblp = dblp_model(retrain)
    create_ce_and_graphs_to_heterodataset(datadblp, modeldblp, 'DBLP', targetdblp, cat_to_explain = -1,  graphs_per_ce = 16, iterations = iterations, compute_acc=False, random_seed = random_seed)
    
    
# ---------------------- evaluation HeteroBAShapes
if run_BAShapes:
    bashapes = create_hetero_ba_houses(500,100) # TODO: Save BAShapes to some file, code already somewhere
    delete_files_in_folder('content/plots/HeteroBAShapes')
    # train GNN 2_hop
    modelHeteroBSM = bsm.train_GNN(True, bashapes, layers=2)
    create_ce_and_graphs_to_heterodataset(bashapes, modelHeteroBSM, 'HeteroBAShapes_BA2hop', '3', cat_to_explain = -1,  graphs_per_ce = 16, iterations = iterations, compute_acc=True, random_seed = random_seed)
    # train GNN 4_hop
    modelHeteroBSM = bsm.train_GNN(True, bashapes, layers=4)
    create_ce_and_graphs_to_heterodataset(bashapes, modelHeteroBSM, 'HeteroBAShapes_BA4hop', '3', cat_to_explain = -1,  graphs_per_ce = 16, iterations = iterations, compute_acc=True, random_seed = random_seed)