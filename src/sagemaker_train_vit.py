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
import torch.optim.lr_scheduler as lr_scheduler




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

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './train'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL', './val'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', './test'))

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
    train_data = AestheticDataset(args.train, transform)
    val_data = AestheticDataset(args.val, transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    print("Datasets loaded")

    print("Total training batches: ", len(train_loader))
    print("Total validation batches: ", len(val_loader))

    if os.path.exists(args.test):
        print("Test directory found.")
        test_data = AestheticDataset(args.test, transform)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    else:
        print("Test directory not found.")

    # Load the pretrained ViT model
    print("Loading ViT model...")
    vit_model = create_model('vit_base_patch16_224', pretrained=True)

    # Modify the output layer of the model
    num_ftrs = vit_model.head.in_features
    vit_model.head = nn.Linear(num_ftrs, 1)    
        
    # Detect if we have a GPU available and if multiple GPUs are available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        vit_model = nn.DataParallel(vit_model)

    # Move the model to the appropriate device
    vit_model = vit_model.to(device)
    print("Model loaded to device: ", device)

    # Set the loss function, optimizer, and learning rate
    loss_function = nn.MSELoss().to(device) # Move loss function to the correct device
    optimizer = optim.Adam(vit_model.parameters(), lr=args.lr) 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    # Define a path for saving/loading checkpoints
    checkpoint_dir = '/opt/ml/checkpoints'  # Amazon SageMaker writes checkpoint data into this directory
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
        for batch_index, (images, labels) in enumerate(train_loader):
            print(f"Training: epoch {epoch+1}, batch {batch_index+1}/{len(train_loader)}")
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
            for batch_index, (images, labels) in enumerate(val_loader):
                print(f"Validating: epoch {epoch+1}, batch {batch_index+1}/{len(val_loader)}")
                images, labels = images.to(device), labels.to(device)
                predictions = vit_model(images).squeeze()
                loss = loss_function(predictions, labels)
                val_loss += loss.item()

        # Print epoch losses
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

        # Update the learning rate
        scheduler.step(val_loss)

        # Save a checkpoint after each epoch
        print("Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': vit_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, os.path.join(checkpoint_dir, f'epoch_{epoch}_checkpoint.pth'))

    # Save the final model to the output directory specified by SageMaker
    print("Saving final model...")
    torch.save(vit_model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    print("Training complete.")
