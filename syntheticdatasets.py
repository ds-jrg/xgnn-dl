from datasets import GenerateRandomGraph
from datasets import GraphMotifAugmenter, HeteroBAMotifDataset


class SyntheticDatasets():
    def __init__(self) -> None:
        pass

    @staticmethod
    def new_dataset_house(num_nodes, num_motifs=None, num_edges=3):
        assert isinstance(num_nodes, int), (num_nodes, type(num_nodes))
        if num_motifs is None:
            num_motifs = num_nodes//5
        # test the new datasets
        # create BA Graph
        ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(
            num_nodes=num_nodes, num_edges=3)
        motif_house_letters = {
            'labels': ['A', 'B', 'B', 'C', 'C'],
            'edges': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
        }
        type_to_classify = 'B'
        synthetic_graph_class = GraphMotifAugmenter(
            motif=motif_house_letters,
            num_motifs=num_motifs,
            orig_graph=ba_graph_nx,
        )
        synthetic_graph = synthetic_graph_class.graph
        # Workaround, fix later

        dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
        dataset_class.augmenter = synthetic_graph_class
        dataset = dataset_class._convert_labels_to_node_types()

        return dataset, dataset_class
