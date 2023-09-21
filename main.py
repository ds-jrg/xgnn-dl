# import GNN_playground #this is only locally important
# print('GNN_playground durchgelaufen')
import bashapes_model as bsm
from datasets import create_hetero_ba_houses, initialize_dblp
from generate_graphs import get_number_of_hdata, get_gnn_outs
from ce_generation import create_graphdict_from_ce, create_test_ce_3011, create_test_ce_3012, generate_cedict_from_ce
from ce_generation import create_random_ce_from_BAHetero, remove_front
from create_random_ce import random_ce_with_startnode, get_graph_from_ce, mutate_ce, length_ce, length_ce, fidelity_ce_testdata, replace_property_of_fillers
from visualization import visualize_best_ces, visualize_hd
from evaluation import ce_score_fct, ce_confusion_iterative, ce_fidelity
import torch
import statistics
# from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
import torch as th
import pickle
import os
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
lambdaone = 0.2  # controls the length of the CE
lambdatwo = 0.0  # controls the variance in the CE output


# ----------------  utils


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

    zeros_indices = random.sample(ones_indices, len(ones_indices) - num_ones_to_keep)

    for index in zeros_indices:
        numbers[index] = 0

    return numbers


def top_unique_scores(results, num_top_results):
    # Sort the list of dictionaries by 'score' in descending order
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    unique_dicts = list()
    seen_scores = set()
    for d in sorted_results:
        score = d['score']
        if score not in seen_scores:
            unique_dicts.append(d)
            seen_scores.add(score)
            if len(unique_dicts) >= 5:
                break
    return unique_dicts


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
        node_types.append(OWLClass(IRI(NS, nt)))
    list_results = []
    for _ in range(number_of_ces):
        local_dict_results = dict()
        ce_here = random_ce_with_startnode(length=start_length, typestart=OWLClass(
            IRI(NS, target_class)), list_of_classes=node_types, list_of_edge_types=edge_ces)
        local_dict_results['CE'] = ce_here
        list_graphs = get_number_of_hdata(ce_here, hd, num_graph_hdata=number_of_graphs)
        local_dict_results['GNN_outs'] = list()
        for graph in list_graphs:
            gnn_out = get_gnn_outs(graph, model, -1)
            local_dict_results['GNN_outs'].append(gnn_out)
        score = ce_score_fct(ce_here, local_dict_results['GNN_outs'], lambdaone, lambdatwo)
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
            local_dict_results['GNN_outs'] = list()
            for graph in list_graphs:
                # generate output of GNN
                # generate the results for acc, f, ...
                gnn_out = get_gnn_outs(graph, model, target_class)
                local_dict_results['GNN_outs'].append(gnn_out)
            score = ce_score_fct(ce_here, local_dict_results['GNN_outs'], lambdaone, lambdatwo)
            local_dict_results['score'] = score
            new_list.append(local_dict_results)
        list_results = sorted(list_results + new_list, key=lambda x: x['score'], reverse=True)[:number_of_ces]
        print(f'Round {_} of {end_length} is finised')
    # only give as feedback the best different top results. Check this, by comparing different scores.
    # In any case, give as feedback 5 CEs, to not throw any errors in the code
    return list_results


def calc_fid_acc_top_results(list_results, model, target_class, dataset):
    if isinstance(target_class, OWLClassExpression):
        target_class = remove_front(target_class.to_string_id())
    for rdict in list_results:
        fidelity = fidelity_ce_testdata(datasetfid=dataset, modelfid=model,
                                        ce_for_fid=rdict['CE'], node_type_expl=target_class, label_expl=-1)

        accuracy = ce_confusion_iterative(rdict['CE'], dataset, [target_class, 0])
        rdict['fidelity'] = fidelity
        rdict['accuracy'] = accuracy
    return list_results


def output_top_results(list_results, target_class, list_all_nt):
    # here we visualize with visualize_hd(hd_graph, addname_for_save, list_all_nodetypes, label_to_explain, add_info='', name_folder='')
    for ce_result in list_results:
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
            '\n' + 'Accuracy: ' + str(ce_result['accuracy'])
        # then, we create the fitting name for savings
        # in the form CE_x_Gr_y
        # where x is the number of the CE and y is the number of the graph
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
    if save_ces:
        list_results = beam_search(hd, model, target_class, start_length, end_length,
                                   number_of_ces, number_of_graphs, num_top_results)
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


# ----------------- Test, to create a test CE, and a test graph.
class_3 = OWLClass(IRI(NS, '3'))
class_2 = OWLClass(IRI(NS, '2'))
class_1 = OWLClass(IRI(NS, '1'))
class_0 = OWLClass(IRI(NS, '0'))
edge = OWLObjectProperty(IRI(NS, 'to'))
test_ce = random_ce_with_startnode(6, class_0, [class_0, class_1], [edge])
graph_to_test_ce = create_graphdict_from_ce(test_ce, ['0', '1'], ['to'], [(
    '0', 'to', '1'), ('1', 'to', '0'), ('0', 'to', '0'), ('1', 'to', '1')])
# print(graph_to_test_ce)


# ---------------------- evaluation DBLP
# this does not work accordingly
random_seed = 3006
if run_DBLP:
    delete_files_in_folder('content/plots/DBLP')
    datadblp, targetdblp = initialize_dblp()
    retrain = False  # set to True, if Dataset should be retrained.
    modeldblp = dblp_model(retrain)
    # create_ce_and_graphs_to_heterodataset(datadblp, modeldblp, 'DBLP', targetdblp, cat_to_explain=-1,
    #                                     graphs_per_ce=16, iterations=iterations, compute_acc=False)


# ---------------------- evaluation HeteroBAShapes
if run_BAShapes:
    bashapes = create_hetero_ba_houses(500, 100)  # TODO: Save BAShapes to some file, code already somewhere
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
#import GNN_playground #this is only locally important
#print('GNN_playground durchgelaufen')
import bashapes_model as bsm
from datasets import create_hetero_ba_houses, initialize_dblp 
from graph_generation import create_graphs_for_heterodata, add_features_and_predict_outcome, compute_accu, compute_prediction_ce, compute_confusion_for_ce_line, compute_f1
from ce_generation import create_graphdict_from_ce, length_ce, create_test_ce_3011, create_test_ce_3012, generate_cedict_from_ce
from ce_generation import create_random_ce_from_BAHetero, remove_front, create_random_ce
from create_random_ce import random_ce_with_startnode
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
    run_DBLP = False
    run_BAShapes = False
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


# ----------------- Test, to create a test CE, and a test graph.
class_3 = OWLClass(IRI(NS,'3'))
class_2 = OWLClass(IRI(NS,'2'))
class_1 = OWLClass(IRI(NS,'1'))
class_0 = OWLClass(IRI(NS,'0'))
edge = OWLObjectProperty(IRI(NS, 'to'))
test_ce = random_ce_with_startnode(6, class_0, [class_0, class_1], [edge])
graph_to_test_ce = create_graphdict_from_ce(test_ce, ['0','1'], ['to'], [('0', 'to', '1'), ('1', 'to', '0'), ('0', 'to', '0'), ('1', 'to', '1')])
print(graph_to_test_ce)




    
    
    

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
    create_ce_and_graphs_to_heterodataset(bashapes, mo