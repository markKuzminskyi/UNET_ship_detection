import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import yaml

import utils
import dataset
from nn_model import UNET

# Downloading config file
config_path = "config.yml"
with open(config_path, "r") as config_file:
    conf = yaml.safe_load(config_file)


DEVICE = conf["device"] if torch.cuda.is_available() else "cpu"

train_ds, test_ds = dataset.get_data()

train_dl = DataLoader(train_ds, batch_size=conf["batch_size"], shuffle=True, pin_memory=torch.cuda.is_available(),
                      num_workers=conf["num_workers"])
val_dl = DataLoader(test_ds, batch_size=conf["batch_size"], shuffle=False, pin_memory=torch.cuda.is_available(),
                    num_workers=conf["num_workers"])


# Training loop, obviously
def train_loop(loader, model, optimizer, loss_fn):
    pbar = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(pbar):
        data = data.to(device=DEVICE)
        target = targets.float().unsqueeze(1).to(device=DEVICE)
        optimizer.zero_grad()

        # forward pass
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            predictions = model(data)
            loss = loss_fn(predictions, target)

        loss.backward()
        optimizer.step()

        # update tqdm progressbar
        pbar.set_postfix(loss=loss.item())


def main():
    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=conf["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    loader = train_dl

    for epoch in range(conf["num_epochs"]):
        train_loop(loader, model, optimizer, loss_fn)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        utils.save_checkpoint(checkpoint)

        # Check accuracy
        utils.check_accuracy(val_dl, model, device=DEVICE)


if __name__ == "__main__":
    main()
