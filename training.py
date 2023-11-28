import shutil

from multiprocessing.dummy import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from skimage.morphology import binary_opening, disk, label

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import tqdm
import yaml

import dataset_preparation as dp
from nn_model import UNET

# Downloading config file
config_path = "config.yml"
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)


DEVICE = config["device"] if torch.cuda.is_available() else "cpu"



def training(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, target)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm progressbar
        loop.set_postfix(loss=loss.item())


def main():
    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    loss = nn.BCEWithLogitsLoss()
    optimizator = optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_ds, test_ds = dp.get_data()

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config["num_epochs"]):
        pass


if __name__ == "__main__":
    main()
