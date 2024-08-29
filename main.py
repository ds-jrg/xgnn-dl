# This File does the experiments and saves them to content/
from beamsearch import BeamSearch
from models import GNN_datasets
from syntheticdatasets import SyntheticDatasets
from bashapes_model import train_GNN
from evaluation import FidelityEvaluator
from main_utils import create_gnn_and_dataset

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
beam_search = BeamSearch(gnn_cl.model, dataset)
beam = beam_search.beam_search()


# evaluation
fideval = FidelityEvaluator(gnn_cl, dataset_class)
