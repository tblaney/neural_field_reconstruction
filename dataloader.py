import os

import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
from utils import download_data

class MeshDataset(Dataset):
    def __init__(self, mesh_path, device=torch.device('cuda:0'), subset="train"):
        """
        Initialize the dataset with the path to the mesh file, device configuration, subset selection,
        and debug mode.

        Parameters:
            mesh_path (str): Path to the mesh file.
            device (torch.device): Device on which tensors will be created.
            subset (str): Specify whether to load the 'train' or 'validation' part of the dataset.
            debug (bool): If True, also load colors data for debugging purposes.
        """

        self.device = device
        self.subset = subset

        # Load data
        self.points, self.labels = self.load_data(mesh_path)
        total_points = self.shuffle_data()
        self.split_data(total_points)


    def load_data(self, mesh_path):
        """
        Load the mesh data from the file and extract vertices as points and colors for labeling.

        Parameters:
            mesh_path (str): Path to the mesh file.

        Returns:
            points, labels: Tensors representing vertices, their labels.
        """

        pcd = trimesh.load(mesh_path)

        points = torch.tensor(pcd.vertices, dtype=torch.float32, device=self.device)
        colors = torch.tensor(pcd.visual.vertex_colors, dtype=torch.float32, device=self.device)[:, :3] / 255.

        red_threshold = torch.tensor([1, 0, 0], device=self.device)
        green_threshold = torch.tensor([0, 1, 0], device=self.device)

        self.points = points
        self.labels = torch.zeros(points.shape[0], dtype=torch.float32, device=self.device)
        self.labels[(colors == green_threshold).all(dim=1)] = 1
        self.labels[(colors == red_threshold).all(dim=1)] = 0

        return self.points, self.labels
    
    def shuffle_data(self):
        """
        Shuffle the data to randomize the order of samples.

        Returns:
            int: Total number of points (samples) after shuffling.
        """

        total_points = self.points.shape[0]
        permutation = torch.randperm(total_points)
        self.points = self.points[permutation]
        self.labels = self.labels[permutation]

        return total_points
    
    def split_data(self, total_points):
        """
        Split the data into training or validation subsets based on the specified subset type.

        Parameters:
            total_points (int): Total number of points in the dataset.
        """
        
        if self.subset == 'train':
            indices = torch.arange(0, int(0.8 * total_points))
        else:
            indices = torch.arange(int(0.8 * total_points), total_points)

        # Apply the indices to subset the data.
        self.points = self.points[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx=None):
        if idx is None:
            idx = torch.randint(0, self.points.shape[0], (1,)).item()

        return self.points[idx], self.labels[idx]


if __name__ == '__main__':
    download_data()
