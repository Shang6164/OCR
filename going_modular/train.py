import torch
from torch.utils.data import DataLoader, Dataset
from model_builder import CenterNetMobileNetV2
from engine import train_one_epoch, evaluate, save_model

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, annotation_path = self.data[idx]
        # Load image and annotation here
        # For simplicity, returning dummy tensors
        image = torch.randn(3, 224, 224)  # Dummy image tensor
        target = torch.zeros(1, 224, 224)  # Dummy target tensor
        if self.transform:
            image = self.transform(image)
        return image, target

if __name__ == "__main__":
    from data_setup import load_data, split_data

    # Load and split data
    IMAGE_DIR = "../card_cropping_data/images"
    ANNOTATION_DIR = "../card_cropping_data/annotations"
    data = load_data(IMAGE_DIR, ANNOTATION_DIR)
    train_data, test_data = split_data(data)

    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNetMobileNetV2(num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    save_model(model, "centernet_mobilenetv2.pth")