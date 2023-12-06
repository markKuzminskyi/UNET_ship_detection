import torch
import torchvision


# Function thst saves current state of model params and optimizer
def save_checkpoint(state, filename="train_checkpoint.pth"):
    print("Saving checkpoint...")
    torch.save(state, filename)


# Function that loads saved model params and other stuff
def load_checkpoint(checkpoint, model):
    print("Loading latest checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])


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


# Saving our predictions to a folder
def save_predictions_as_imgs(loader, model, folder="/saved_imgs", device="cpu"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}")

        model.train()
