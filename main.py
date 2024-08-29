# This File does the experiments and saves them to content/
import random
from beamsearch import BeamSearch
from models import GNN_datasets
from syntheticdatasets import SyntheticDatasets
from bashapes_model import train_GNN
from evaluation import FidelityEvaluator
from main_utils import create_gnn_and_dataset, create_test_dataset
from owlapy.render import DLSyntaxObjectRenderer
dlsr = DLSyntaxObjectRenderer()

# setup for parameters

retrain_GNN_and_data = False


# ------- Code -----------
gnn_cl, dataset, dataset_class = create_gnn_and_dataset('house_1000',
                                                        'SAGE_2_20',
                                                        gnn_epochs=20,
                                                        gnn_layers=2,
                                                        type_to_classify='B',
                                                        retrain=retrain_GNN_and_data,
                                                        )


# beam search CEs
beam_search = BeamSearch(gnn_cl.model, dataset,
                         beam_width=100, beam_depth=12, max_depth=2)
beam = beam_search.beam_search()


# return top CEs
test_ce_dataset = create_test_dataset(num_nodes=100)
fideval = FidelityEvaluator(test_ce_dataset, gnn_cl.model, type_to_explain='B')

for i in range(10):
    acc = fideval.score_fid_accuracy(beam[i])
    print(
        f"Number {i+1} CE is {dlsr.render(beam[i])} has an acc of {round(acc, 2)}")
