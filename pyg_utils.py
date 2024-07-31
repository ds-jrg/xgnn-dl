import torch
from torch_geometric.data import HeteroData


class PyGUtils():
    """
    This class summarizes all functions to manipulate PyG graphs.
    Currently:
    - Add Edge: add_edge_to_hdata
    - Make a PyG graph bi-directed: make_hdata_bidirected
    """

    @staticmethod
    def add_edge_to_hdata(hetero_graph, start_type, edge_type, end_type, start_id: int, end_id: int):
        start_id_tensor = torch.tensor([start_id], dtype=torch.long)
        end_id_tensor = torch.tensor([end_id], dtype=torch.long)
        start_id, end_id = int(start_id), int(end_id)
        assert isinstance(
            hetero_graph, HeteroData), "The graph is not a heterogenous graph."

        if (start_type, edge_type, end_type) in hetero_graph.edge_types:
            list_ids_start, list_ids_end = [row.tolist()
                                            for row in hetero_graph[(start_type, edge_type, end_type)].edge_index]
            list_ids_start.append(start_id)
            list_ids_end.append(end_id)
            hetero_graph[(start_type, edge_type, end_type)].edge_index = torch.tensor(
                [list_ids_start, list_ids_end])
            changes_made = True
        else:
            hetero_graph[start_type, edge_type, end_type].edge_index = torch.tensor([
                                                                                    [start_id], [end_id]])
            changes_made = True

        return hetero_graph

    @staticmethod
    def make_hdata_bidirected(hetero_graph):
        """
        This makes a heterogenous graph bidirected and checks on validity: Each edge should exist in 2 directions.
        """
        for edge_type in hetero_graph.edge_types:
            start_type, relation_type, end_type = edge_type

            # Get the edge indices for this type
            edge_indices = hetero_graph[start_type,
                                        relation_type, end_type].edge_index
            start_indices, end_indices = edge_indices[0], edge_indices[1]

            # Iterate through the edges
            for start_id, end_id in zip(start_indices, end_indices):
                # Check if the reverse edge exists
                if (end_type, relation_type, start_type) in hetero_graph.edge_types:
                    reverse_edge_index = hetero_graph[end_type,
                                                      relation_type, start_type].edge_index
                    if not any(end_id == reverse_edge_index[0][i] and start_id == reverse_edge_index[1][i] for i in range(len(reverse_edge_index[0]))):
                        # Add the reverse edge
                        # Assuming a function add_edge_to_hdata as defined previously
                        hetero_graph = GraphLibraryConverter.add_edge_to_hdata(
                            hetero_graph, end_type, relation_type, start_type, end_id.item(), start_id.item())
                else:
                    # Add the reverse edge
                    # Assuming a function add_edge_to_hdata as defined previously
                    hetero_graph = GraphLibraryConverter.add_edge_to_hdata(
                        hetero_graph, end_type, relation_type, start_type, end_id.item(), start_id.item())
        assert isinstance(
            hetero_graph, HeteroData), "The graph is not a heterogenous graph."
        assert hetero_graph.is_undirected(
        ), f"The graph {hetero_graph} is not undirected."
        return hetero_graph
