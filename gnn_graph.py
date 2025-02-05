import torch
from torch_geometric.data import HeteroData

class StrokeGraph(HeteroData):
    def __init__(self, all_edges_matrix, intersection_matrix):
        super(StrokeGraph, self).__init__()

        num_strokes = all_edges_matrix.shape[0]

        # Convert to tensors
        all_edges_matrix = torch.tensor(all_edges_matrix, dtype=torch.float32)  # (num_strokes, 6)
        intersection_matrix = torch.tensor(intersection_matrix, dtype=torch.int64)  # (num_strokes, num_strokes)

        # Store stroke node features
        self['stroke'].x = all_edges_matrix  # Shape: (num_strokes, 6)

        # List to store directed edges
        edge_index = []

        # 1. Add stroke order edges (0 → 1 → 2 → ... → num_strokes-1)
        for i in range(num_strokes - 1):
            edge_index.append([i, i + 1])

        # 2. Add intersection-based edges
        for src in range(num_strokes):
            for dst in range(num_strokes):
                if intersection_matrix[src, dst] == 1 and src != dst:  # Ensure no self-loops
                    edge_index.append([src, dst])

        # Convert edges to tensor and assign
        if edge_index:
            self['stroke', 'connected_to', 'stroke'].edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Store stroke indices (optional)
        self['stroke'].index = torch.arange(num_strokes, dtype=torch.long)
