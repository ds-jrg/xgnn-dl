import os
import torch
from torch_geometric.data import HeteroData
import random
from owlapy.model import IRI
from owlapy.render import DLSyntaxObjectRenderer

dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns

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

    zeros_indices = random.sample(ones_indices, len(ones_indices) - num_ones_to_keep)

    for index in zeros_indices:
        numbers[index] = 0

    return numbers


def top_unique_scores(results, num_top_results):
    """
    Returns the top scores from all evaluated CEs, which are saved in the list results.
    Omits CEs with the same score.
    """
    # Sort the list of dictionaries by 'score' in descending order
    sorted_results = list(sorted(results, key=lambda x: x['score'], reverse=True))
    sorted_results = sorted_results[:num_top_results]
    return sorted_results
