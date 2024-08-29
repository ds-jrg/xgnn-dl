import pickle
import os
from syntheticdatasets import SyntheticDatasets
from models import GNN_datasets


def create_gnn_and_dataset(dataset_name,
                           gnn_name,
                           type_to_classify=None,
                           gnn_epochs=10,
                           retrain=False,
                           gnn='SAGE',
                           gnn_layers=2,
                           dataset='house',  # type of data like house
                           num_nodes=100,
                           ):
    folder_path_ds = 'content/datasets'
    folder_path_gnn = 'content/gnns'
    # create Folders
    os.makedirs(folder_path_ds, exist_ok=True)
    os.makedirs(folder_path_gnn, exist_ok=True)
    data_path = 'content/'+dataset_name+'_'+'dataset.pkl'
    gnn_path = 'content/'+gnn_name+'_'+'gnn.pkl'
    if not retrain:
        try:
            with open(data_path, 'rb') as f:
                dataset, dataset_class = pickle.load(f)
            with open(gnn_path, 'rb') as f:
                gnn_cl = pickle.load(f)
        except Exception:
            retrain = True
    if retrain:
        if dataset is None:
            raise Exception("Dataset is None")
        if gnn is None:
            raise Exception("GNN is None")
        if dataset == 'house':
            SyntheticData = SyntheticDatasets()
            dataset, dataset_class = SyntheticData.new_dataset_house(num_nodes)

        if gnn == 'SAGE':
            gnn_cl = GNN_datasets(data=dataset, type_to_classify='B')
            gnn_cl.train_model(epochs=20)

        # store everything
        data_tuple = (dataset, dataset_class)
        with open(data_path, 'wb') as f:
            pickle.dump(data_tuple, f)
        with open(gnn_path, 'wb') as f:
            pickle.dump(gnn_cl, f)

    return gnn_cl, dataset, dataset_class


def create_test_dataset(dataset='house', num_nodes=100):
    if dataset == 'house':
        SyntheticData = SyntheticDatasets()
        dataset, _ = SyntheticData.new_dataset_house(num_nodes)
    return dataset
