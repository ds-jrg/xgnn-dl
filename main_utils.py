import pickle
import os
from syntheticdatasets import SyntheticDatasets
from models import GNNDatasets


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
    gnn_path = 'content/'+gnn_name+'_'+dataset_name+'_'+'gnn.pkl'
    if not retrain:
        try:
            with open(data_path, 'rb') as f:
                dataset, dataset_class = pickle.load(f)
            with open(gnn_path, 'rb') as f:
                gnn_cl = pickle.load(f)
        except Exception:
            retrain = True
    if retrain:
        if gnn is None:
            raise Exception("GNN is None")
        if dataset_name == 'house':
            dataset, dataset_class = SyntheticDatasets.new_dataset_house(
                num_nodes)
        elif dataset_name == 'circle5':
            dataset, dataset_class = SyntheticDatasets.new_dataset_circle5(
                num_nodes)
        else:
            raise Exception("Dataset is not recognized.")

        if gnn == 'SAGE':
            gnn_cl = GNNDatasets(
                data=dataset, type_to_classify=type_to_classify)
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
        dataset, _ = SyntheticDatasets.new_dataset_house(num_nodes)
    elif dataset == 'circle5':
        dataset, _ = SyntheticDatasets.new_dataset_circle5(num_nodes)
    return dataset
