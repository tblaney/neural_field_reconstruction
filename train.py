import argparse
import os

import trimesh
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MeshDataset
import reconstruction
from model import OCC

from torch.optim import Adam
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR

import json

class OccConfig:
    """
    A configuration class to manage training options and hyperparameters, loaded from a JSON file.
    """
    def __init__(self, name):
        self.name = name
        self.load_config(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_config(self, filename):
        """
        Load configuration from a JSON file.
        """
        try:
            with open(f"{filename}.json", "r") as file:
                data = json.load(file)
                for key, value in data.items():
                    setattr(self, key, value)

        except FileNotFoundError:
            raise FileNotFoundError(f"no configuration file found for the name '{filename}'")
        except json.JSONDecodeError:
            raise ValueError("error decoding JSON")

    def log_config(self):
        """
        Log all configurations of the OccConfig instance.
        """
        print("config settings:")
        for attr, value in self.__dict__.items():
            print(f"  {attr}: {value}")

EPSILON = 1e-15
IS_GOOGLE_COLAB = False

CONFIG_HASH = "hash"
CONFIG_1LOD = "one_lod"
CONFIG_MLOD = "m_lod"

class OccTrainer:
    def __init__(self, config):
        """
        Initializes the Trainer class with specified configuration options.

        Parameters:
        config (OccConfig): Configuration options with training parameters and device settings.
        """
        self.config = config
        self.device = config.device

        self.train_dataset = MeshDataset(self.config.current_obj_path, device=self.device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model = OCC(self.config).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.99), eps=EPSILON, weight_decay=self.config.weight_decay)
        self.criterion = BCELoss()
        self.scheduler = StepLR(self.optimizer, step_size=self.config.lr_decay_step, gamma=self.config.lr_decay_gamma)

        self.num_epochs = self.config.epochs

    def run(self):
        """
        Run training for a specified number of epochs and save the best model based on loss.
        """

        self.model.train()
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            with tqdm(self.train_dataloader, unit="batch") as pbar:
                for points, labels in pbar:
                    pbar.set_description(f"epoch {epoch + 1}")
                    labels = labels.view(-1, 1)

                    # Zero out gradients
                    self.optimizer.zero_grad()

                    # Form predictions
                    pred = self.model(points)

                    # Calculate loss
                    loss = self.criterion(pred, labels)

                    # Calculate gradients
                    loss.backward()

                    # Step and backpropagate
                    self.optimizer.step()
                    self.scheduler.step()

                    total_loss += loss.item()

                    pbar.set_postfix(loss=total_loss / len(self.train_dataloader))

                # Save if necessary
                if total_loss < best_loss:
                    best_loss = total_loss
                    current_obj_name = self.config.current_obj.replace(".obj", "")
                    if IS_GOOGLE_COLAB:
                        torch.save(self.model.state_dict(), f'drive/MyDrive/3dv_hw3/{self.config.name}_{current_obj_name}.pth')
                        print("model saved to: {}".format(f'drive/MyDrive/3dv_hw3/{self.config.name}_{current_obj_name}.pth'))
                    else:
                        torch.save(self.model.state_dict(), f'{self.config.name}_{current_obj_name}.pth')
                    print("final model saved to: {}".format(f'{self.config.name}_{current_obj_name}.pth'))
                elif epoch == self.num_epochs - 1:
                    current_obj_name = self.config.current_obj.replace(".obj", "")
                    torch.save(self.model.state_dict(), f'{self.config.name}_{current_obj_name}_final.pth')
                    print("final model saved to: {}".format(f'{self.config.name}_{current_obj_name}_final.pth'))


    def get_num_params(self):
        """
        Get the number of trainable parameters in the model.

        Returns:
        int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


if __name__ == '__main__':

    config = OccConfig(CONFIG_HASH)

    print(f'load config: {config.name}')

    for obj_file in os.listdir("processed"):

        print("##################")

        print("obj path: {}".format(obj_file))

        config.current_obj_path = "processed/{}".format(obj_file)
        config.current_obj = obj_file

        config.log_config()

        current_obj_name = config.current_obj.replace(".obj", "")

        trainer = OccTrainer(config)

        print('model has {} params!'.format(trainer.get_num_params()))

        print("##################")

        trainer.run()
