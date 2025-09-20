import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run the training on (CPU or GPU).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        criterion: Loss function.
        device: Device to run the evaluation on (CPU or GPU).

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(dataloader)

def save_model(model, path):
    """
    Save the model to the specified path.

    Args:
        model (nn.Module): The model to save.
        path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """
    Load the model from the specified path.

    Args:
        model (nn.Module): The model to load weights into.
        path (str): Path to the saved model.
        device: Device to map the model to (CPU or GPU).
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)