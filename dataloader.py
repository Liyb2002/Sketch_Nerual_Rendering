import os
import json

import cad2sketch_stroke_features

class cad2sketch_dataset_loader:
    def __init__(self):
        """
        Initializes the dataset generator by setting paths and loading the dataset.
        """
        self.data_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch')
        self.idx = 0  # Index for naming processed folders
        self.load_dataset()

    def load_dataset(self):
        """
        Loads the dataset by iterating over all subfolders and processing each valid subfolder.
        """
        if not os.path.exists(self.data_path):
            print(f"Dataset path '{self.data_path}' not found.")
            return

        folders = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]
        
        if not folders:
            print("No folders found in the dataset directory.")
            return

        for folder in folders:
            folder_path = os.path.join(self.data_path, folder)
            subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]

            if not subfolders:
                print(f"No subfolders found in '{folder}'. Skipping...")
                continue
            
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder)
                self.process_subfolder(folder_path, subfolder_path)

    def process_subfolder(self, folder_path, subfolder_path):
        """
        Processes an individual subfolder by reading JSON files and extracting relevant data.
        """
        final_edges_file_path = os.path.join(subfolder_path, 'final_edges.json')
        all_edges_file_path = os.path.join(subfolder_path, 'all_edges.json')
        strokes_dict_path = os.path.join(subfolder_path, 'strokes_dict.json')

        # Check if required JSON files exist
        if not os.path.exists(final_edges_file_path):
            print(f"Skipping {subfolder_path}: 'final_edges.json' not found.")
            return
        if not os.path.exists(all_edges_file_path):
            print(f"Skipping {subfolder_path}: 'all_edges.json' not found.")
            return
        if not os.path.exists(strokes_dict_path):
            print(f"Skipping {subfolder_path}: 'strokes_dict.json' not found.")
            return

        self.idx += 1  # Increment index for the next subfolder

        # Load stroke connection matrix
        strokes_dict_data = self.read_json(strokes_dict_path)

        # Load and visualize all edges
        all_edges_data = self.read_json(all_edges_file_path)
        all_edges_matrix = cad2sketch_stroke_features.simple_build_all_edges_features(all_edges_data)
        # cad2sketch_stroke_features.via_all_edges(all_edges_data)


        # Load and visualize final edges
        final_edges_data = self.read_json(final_edges_file_path)
        final_edges_matrix = cad2sketch_stroke_features.simple_build_final_edges_features(final_edges_data, all_edges_data)
        cad2sketch_stroke_features.vis_final_edges(final_edges_data)

    def read_json(self, file_path):
        """
        Reads a JSON file and returns its contents.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None

