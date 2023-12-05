import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import yaml

import dataset as dp
from nn_model import UNET

# Downloading config file
config_path = "config.yml"
with open(config_path, "r") as config_file:
    conf = yaml.safe_load(config_file)


DEVICE = conf["device"] if torch.cuda.is_available() else "cpu"

train_ds, test_ds = dp.get_data()

train_dl = DataLoader(train_ds, batch_size=conf["batch_size"], shuffle=True, pin_memory=torch.cuda.is_available(),
                      num_workers=conf["num_workers"])
val_dl = DataLoader(test_ds, batch_size=conf["batch_size"], shuffle=False, pin_memory=torch.cuda.is_available(),
                    num_workers=conf["num_workers"])


# Function that calculates accuracy
def check_accuracy(loader, model, device="cpu"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def training(loader, model, optimizer, loss_fn, scaler):
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
    loss_fn = nn.BCEWithLogitsLoss()
    optimizator = optim.Adam(model.parameters(), lr=conf["learning_rate"])

    for epoch in range(conf["num_epochs"]):
        pass


if __name__ == "__main__":
    main()
