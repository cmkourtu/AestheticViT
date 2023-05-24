import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm import create_model


class AestheticDataset(Dataset):
    def __init__(self, dir, transform=None, extensions=("jpg", "jpeg", "png", "bmp", "tiff")):
        self.dir = dir
        self.transform = transform
        self.extensions = extensions
        self.image_paths = sorted([os.path.join(dir, img) for img in os.listdir(dir) if img.lower().endswith(self.extensions)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(self.image_paths[idx].split('/')[-1].split('.')[0]))  # Assumes the label is in the filename
        return image, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)

    # Directories for training, validation and output data
    parser.add_argument('--output_data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, default='')

    args = parser.parse_args()

    print("Arguments: ", args)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the datasets and DataLoaders
    print("Loading datasets...")
    train_data = AestheticDataset(args.train_dir, transform)
    val_data = AestheticDataset(args.val_dir, transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    print("Datasets loaded")

    if os.path.exists(args.test_dir):
        print("Test directory found.")
        test_data = AestheticDataset(args.test_dir, transform)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    else:
        print("Test directory not found.")

    # Load the pretrained ViT model
    print("Loading ViT model...")
    vit_model = create_model('vit_base_patch16_224', pretrained=True)

    # Modify the model for the regression task
    vit_model.head = nn.Linear(vit_model.head.in_features, 1)

    # Set the loss function, optimizer, and learning rate
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(vit_model.parameters(), lr=args.lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model.to(device)
    print("Model loaded to device: ", device)

    # Define a path for saving/loading checkpoints
    checkpoint_dir = os.path.join(args.output_data_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load from the most recent checkpoint if it exists
    latest_checkpoint = max(glob.glob(checkpoint_dir + "/*"), default=None, key=os.path.getctime)
    if latest_checkpoint is not None:
        print(f"Loading from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        start_epoch = checkpoint['epoch']
        vit_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0

    # Training and validation loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Training
        vit_model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = vit_model(images).squeeze()
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        vit_model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = vit_model(images).squeeze()
                loss = loss_function(predictions, labels)
                val_loss += loss.item()

        # Print epoch losses
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

        # Save a checkpoint after each epoch
        print("Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': vit_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, os.path.join(checkpoint_dir, f'epoch_{epoch}_checkpoint.pth'))

    # Save the final model to the output directory specified
    print("Saving final model...")
    torch.save(vit_model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    print("Training complete.")
