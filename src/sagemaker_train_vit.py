import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm import create_model


class AestheticDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(dir, img) for img in os.listdir(dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(self.image_paths[idx].split('/')[-1].split('.')[0])) # Assumes the label is in the filename
        return image, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args = parser.parse_args()

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the datasets and DataLoaders
    train_data = AestheticDataset(args.train, transform)
    val_data = AestheticDataset(args.val, transform)
    test_data = AestheticDataset(args.test, transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Load the pretrained ViT model
    vit_model = create_model('vit_base_patch16_224', pretrained=True)

    # Modify the model for the regression task
    vit_model.head = nn.Linear(vit_model.head.in_features, 1)

    # Set the loss function, optimizer, and learning rate
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(vit_model.parameters(), lr=args.lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model.to(device)

    for epoch in range(args.epochs):
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

    # Save the model to the output directory specified by SageMaker
    torch.save(vit_model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
