from datasets import GenerateRandomGraph
from datasets import GraphMotifAugmenter, HeteroBAMotifDataset


class SyntheticDatasets():
    def __init__(self) -> None:
        pass
    motif_house = {
        'labels': ['A', 'B', 'B', 'C', 'C'],
        'edges': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    }

    motif_circle5 = {'labels': ['A', 'A', 'A', 'A', 'A'],
                     'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]}

    motif_star5 = {
        'labels': ['C', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A'],
        'edges': [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]
    }

    motif_wheel5 = {
        'labels': ['C', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A'],
        'edges': [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 7), (7, 8), (8, 9), (9, 10), (10, 6)]
    }

    @staticmethod
    def new_dataset_motif(num_nodes, motif, num_motifs=None, num_edges=3):
        assert isinstance(num_nodes, int), (num_nodes, type(num_nodes))
        assert isinstance(motif, dict), (motif, type(motif))
        if num_motifs is None:
            num_motifs = num_nodes//5
        # test the new datasets
        # create BA Graph
        ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(
            num_nodes=num_nodes, num_edges=num_edges)
        type_to_classify = 'A'
        synthetic_graph_class = GraphMotifAugmenter(
            motif=motif,
            num_motifs=num_motifs,
            orig_graph=ba_graph_nx,
        )
        synthetic_graph = synthetic_graph_class.graph
        # Workaround, fix later
        dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
        dataset_class.augmenter = synthetic_graph_class
        dataset = dataset_class._convert_labels_to_node_types()

        return dataset, dataset_class

    @staticmethod
    def new_dataset_house(num_nodes, num_motifs=None, num_edges=3):
        assert isinstance(num_nodes, int), (num_nodes, type(num_nodes))
        if num_motifs is None:
            num_motifs = num_nodes//5
        # test the new datasets
        # create BA Graph
        ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(
            num_nodes=num_nodes, num_edges=num_edges)
        type_to_classify = 'B'
        synthetic_graph_class = GraphMotifAugmenter(
            motif=SyntheticDatasets.motif_house,
            num_motifs=num_motifs,
            orig_graph=ba_graph_nx,
        )
        synthetic_graph = synthetic_graph_class.graph
        # Workaround, fix later

        dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
        dataset_class.augmenter = synthetic_graph_class
        dataset = dataset_class._convert_labels_to_node_types()

        return dataset, dataset_class

    @staticmethod
    def new_dataset_circle5(num_nodes, num_motifs=None, num_edges=3):
        assert isinstance(num_nodes, int), (num_nodes, type(num_nodes))
        if num_motifs is None:
            num_motifs = num_nodes//5
        # test the new datasets
        # create BA Graph
        ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(
            num_nodes=num_nodes, num_edges=num_edges)
        type_to_classify = 'A'
        synthetic_graph_class = GraphMotifAugmenter(
            motif=SyntheticDatasets.motif_circle5,
            num_motifs=num_motifs,
            orig_graph=ba_graph_nx,
        )
        synthetic_graph = synthetic_graph_class.graph
        # Workaround, fix later

        dataset_class = HeteroBAMotifDataset(synthetic_graph, type_to_classify)
        dataset_class.augmenter = synthetic_graph_class
        dataset = dataset_class._convert_labels_to_node_types()
        return dataset, dataset_class
