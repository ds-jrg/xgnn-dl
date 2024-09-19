import unittest
from models import HeteroRGCN, GNNDatasets
from syntheticdatasets import SyntheticDatasets


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        SyntheticData = SyntheticDatasets()
        self.dataset, self.dataset_class = SyntheticData.new_dataset_house(
            1000)

        self.rgcn_cl = HeteroRGCN(
            self.dataset, num_relations=1, num_nodefeatures=1, num_classes=2)
        self.gnn_cl = GNNDatasets(
            self.dataset, num_layers=2, type_to_classify='B', model=self.rgcn_cl)
        self.gnn_cl.train_model(epochs=20)

    def test_model(self):
        print('yeahyy testing')
        print(self.dataset.metadata())

        self.assertIsInstance(self.rgcn_cl.model, HeteroRGCN)


if __name__ == '__main__':
    unittest.main()
