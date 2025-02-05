import sys
import torch
from tqdm import tqdm
from dataloader import cad2sketch_dataset_loader
from torch.utils.data import DataLoader

import gnn_graph

# Initialize dataset
dataset = cad2sketch_dataset_loader()

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Iterate through the dataset
for data in tqdm(dataloader, desc="Building Graphs"):
    intersection_matrix, all_edges_matrix, final_edges_matrix = data
    

    all_edges_matrix = all_edges_matrix.squeeze(0)  # Shape: (num_strokes, 6)
    intersection_matrix = intersection_matrix.squeeze(0)  # Shape: (num_strokes, num_strokes)

    cur_gnn_graph = gnn_graph.StrokeGraph(all_edges_matrix, intersection_matrix)
    print("Node features (stroke):", cur_gnn_graph['stroke'].x.shape)  # Should be (num_strokes, 6)
