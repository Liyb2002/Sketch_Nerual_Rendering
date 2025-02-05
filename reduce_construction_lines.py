import sys
import torch
from tqdm import tqdm
from dataloader import cad2sketch_dataset_loader
from torch.utils.data import DataLoader

# Initialize dataset
dataset = cad2sketch_dataset_loader()

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Iterate through the dataset
for data in tqdm(dataloader, desc="Building Graphs"):
    intersection_matrix, all_edges_matrix, final_edges_matrix = data
    
    # print(f"Intersection: {intersection_matrix.shape}, All Edges: {all_edges_matrix.shape}, Final Edges: {final_edges_matrix.shape}")
