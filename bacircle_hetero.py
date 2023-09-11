import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif
from sklearn.model_selection import train_test_split
import random
from collections import Counter

def count_ints_total(input_list, intput):
    count = 0
    for element in input_list:
        if element == intput:
            count += 1
    return count
def count_ints_until_entry(input_list, intput, entry):  #works
    return count_ints_total(input_list[:entry], intput)


#utils: retrieve the second argument of the list_current_to_new_indices:
def new_index(list_of_pairs, index):
    for pair in list_of_pairs:
        if pair[0] == index:
            return pair[1]

import random
def replace_random_zeros_with_one_or_three(input_list, prob_replace=0.07):
    output_list = []
    label_list = []
    counter = Counter(input_list)

    # Iterate over the counter and print the values and their frequencies
    for index_value in range(len(input_list)):
        if input_list[index_value] == 1:
            input_list[index_value] = 2
        elif input_list[index_value] == 2:
            input_list[index_value] = 1
    
    for value in input_list:
        if value == 0 and random.random() < 2*prob_replace:
            output_list.append(3)
            label_list.append(0)
        elif value == 0 and random.random() < prob_replace:
            output_list.append(1)
        elif value == 0 and random.random() < prob_replace:
            output_list.append(2)
        else:
            output_list.append(value)
            if value == 3:
                label_list.append(1)
    counter = Counter(input_list)
    # Iterate over the counter and print the values and their frequencies
    for value, frequency in counter.items():
        print(f"Value: {value}, Frequency: {frequency}")
    return output_list, label_list



# it creates houses with labels 3-2-1 (top->bottom)
def create_hetero_ba_cycles(not_house_nodes, houses):
    dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=25, num_edges=2),
    motif_generator=CycleMotif(5),
    num_motifs=1,
    num_graphs=500,
)
    print(dataset)
    homgraph = dataset.get_graph()
    listnodetype = homgraph.y.tolist()
    listedgeindex = homgraph.edge_index.tolist()
    print(homgraph)
    #randomly change some nodes of type 0 to type 3 or 1 and also retrieve a list of labels for nodes of type '3'
    #listnodetype, label_list = replace_random_zeros_with_one_or_three(listnodetype, 0.05)  #replace with 1,2,3,4,5
    #number_of_each_type = []