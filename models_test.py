import unittest
from models import HeteroRGCN, GNNDatasets, RGCN, RGCNPreProcessor
from syntheticdatasets import SyntheticDatasets


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        SyntheticData = SyntheticDatasets()
        self.dataset, self.dataset_class = SyntheticData.new_dataset_house(
            500)

        self.rgcn_cl = RGCN(
            self.dataset, num_relations=16, num_bases=3, hidden_layers=6, out_channels=2)  # TODO: Rename in hidden_channels
        self.gnn_cl = RGCNPreProcessor(
            self.dataset, 'B')
        self.gnn_cl.train_model(epochs=60)

    def test_model(self):
        print('yeahyy testing')


if __name__ == '__main__':
    unittest.main()
