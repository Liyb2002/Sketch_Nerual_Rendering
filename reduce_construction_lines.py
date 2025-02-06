import sys
import torch
from tqdm import tqdm
from dataloader import cad2sketch_dataset_loader
from torch.utils.data import DataLoader

import gnn_graph

import gnn.gnn


graph_encoder = gnn.gnn.SemanticModule()

# Initialize dataset
dataset = cad2sketch_dataset_loader()

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

graphs = []
final_edges_mask = []

# Iterate through the dataset
for data in tqdm(dataloader, desc="Building Graphs"):
    intersection_matrix, all_edges_matrix, final_edges_matrix = data
    

    all_edges_matrix = all_edges_matrix.squeeze(0)  # Shape: (num_strokes, 6)
    intersection_matrix = intersection_matrix.squeeze(0)  # Shape: (num_strokes, num_strokes)

    cur_gnn_graph = gnn_graph.StrokeGraph(all_edges_matrix, intersection_matrix)

    graphs.append(cur_gnn_graph)
    final_edges_mask.append(final_edges_matrix)

    if len(graphs) > 10:
        break


split_index = int(0.8 * len(graphs))
train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
train_masks, val_masks = final_edges_mask[:split_index], final_edges_mask[split_index:]


# Training and validation loop
epochs = 30  # Number of epochs
best_accuracy = 0.0

for epoch in range(epochs):
    for graph, mask in zip(train_graphs, train_masks):
        x_dict = graph_encoder(graph.x_dict, graph.edge_index_dict)
        print("x_dict", x_dict['stroke'].shape)

