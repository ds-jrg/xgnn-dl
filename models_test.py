import unittest
from models import HeteroRGCN, GNNDatasets, RGCN, RGCN_train
from syntheticdatasets import SyntheticDatasets


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        SyntheticData = SyntheticDatasets()
        self.dataset, self.dataset_class = SyntheticData.new_dataset_house(
            100)

        self.rgcn_cl = RGCN(
            self.dataset, num_relations=16, num_bases=20, hidden_layers=16, out_channels=2)
        self.gnn_cl = RGCN_train(
            self.dataset, 'B')
        self.gnn_cl.train_model(epochs=20)

    def test_model(self):
        print('yeahyy testing')


if __name__ == '__main__':
    unittest.main()
