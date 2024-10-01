from datasets import GenerateRandomGraph
from datasets import GraphMotifAugmenter, HeteroBAMotifDataset


class SyntheticDatasets():
    def __init__(self) -> None:
        pass
    motif_house = {
        'labels': ['A', 'B', 'B', 'C', 'C'],
        'edges': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    }

    motif_circle = {'labels': ['A', 'A', 'A', 'A', 'A'],
                    'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]}

    motif_star = {
        'labels': ['C', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A'],
        'edges': [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]
    }

    motif_wheel = {
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
    def new_dataset_n_motif(num_nodes, motifs: dict, num_motifs=None, num_edges=3):
        if num_motifs is None:
            num_motifs = num_nodes//5

        total_motifs = sum(len(motifs.values())
                           for motifs in motifs)
        # could be in total less than numn_motifs
        num_onemotif = num_motifs//total_motifs
        ba_graph_nx = GenerateRandomGraph.create_BAGraph_nx(
            num_nodes=num_nodes, num_edges=num_edges)
        synthetic_graph_class = GraphMotifAugmenter(
            motif=motifs['positive'],
            num_motifs=num_onemotif,
            orig_graph=ba_graph_nx,
        )
        synthetic_graph_class.add_n_motifs_negative(
            motifs['negative'])
