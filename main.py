# This File does the experiments and saves them to content/
import random
from beamsearch import BeamSearch
from models import GNNDatasets
from syntheticdatasets import SyntheticDatasets
from evaluation import FidelityEvaluator
from main_utils import create_gnn_and_dataset, create_test_dataset
from owlapy.render import DLSyntaxObjectRenderer
dlsr = DLSyntaxObjectRenderer()

# setup for parameters

retrain_GNN_and_data = True
list_datasets = ['house', 'circle', 'star', 'wheel',
                 [{'positive': ['house', 'wheel'], 'negative': ['circle',  'star']}],
                 ]
dict_types_to_classify = {'house': 'A',
                          'circle': 'A', 'star': 'A', 'wheel': 'A'}
gnn_parameters = [{'name': 'SAGE_2_50', 'gnn_layers': 4, 'epochs': 50},
                  ]


# ------- Code -----------
gnns = dict()
data = dict()
data_cl = dict()

for ds in list_datasets:
    for gnnparams in gnn_parameters:
        gnns[ds], data[ds], data_cl[ds] = create_gnn_and_dataset(dataset_name=ds,
                                                                 gnn_name=gnnparams['name'],
                                                                 gnn_epochs=gnnparams['epochs'],
                                                                 gnn_layers=gnnparams['gnn_layers'],
                                                                 type_to_classify='A',  # default: A
                                                                 retrain=retrain_GNN_and_data
                                                                 )

        # beam search CEs
        beam_search = BeamSearch(gnns[ds].model,
                                 data[ds],
                                 beam_width=500,
                                 beam_depth=15,
                                 # max_depth of created CEs, should be number of GNN layers
                                 max_depth=gnnparams['gnn_layers'],
                                 )
        beam = beam_search.beam_search()

        # return top CEs
        test_ce_dataset = create_test_dataset(dataset_name=ds, num_nodes=1000)
        fideval = FidelityEvaluator(
            test_ce_dataset, gnns[ds].model, type_to_explain=dict_types_to_classify[ds])
        print(f"Top 10 CEs: of dataset {ds} and gnn {gnnparams['name']}")
        for i in range(10):

            acc = fideval.score_fid_accuracy(beam[i])
            print(
                f"Number {i+1} CE is {dlsr.render(beam[i])} and has an acc of {round(acc, 2)}")
