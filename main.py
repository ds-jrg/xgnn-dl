# This File does the experiments and saves them to content/
from beamsearch import BeamSearch
import models
from syntheticdatasets import SyntheticDatasets
from bashapes_model import train_GNN

SyntheticData = SyntheticDatasets()
dataset, dataset_class = SyntheticData.new_dataset_house(100)
print(dataset)

# gnn on dataset
model_on_sdata = train_GNN(True, dataset, 2)
