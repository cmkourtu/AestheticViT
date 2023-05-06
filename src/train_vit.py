import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm import create_model

# Create a custom dataset class
class AestheticDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return image, label

# Load the dataset and preprocess it
# Replace these with the appropriate file paths and labels for your dataset
train_image_paths, val_image_paths, test_image_paths = ..., ..., ...
train_labels, val_labels, test_labels = ..., ..., ...

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the datasets and DataLoaders
train_data = AestheticDataset(train_image_paths, train_labels, transform)
val_data = AestheticDataset(val_image_paths, val_labels, transform)
test_data = AestheticDataset(test_image_paths, test_labels, transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load the pretrained ViT model
vit_model = create_model('vit_base_patch16_224', pretrained=True)

# Modify the model for the regression task
vit_model.head = nn.Linear(vit_model.head.in_features, 1)

# Set the loss function, optimizer, and learning rate
loss_function = nn.MSELoss()
optimizer = optim.Adam(vit_model.parameters(), lr=1e-4)

# Train the model on the prepared dataset
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit_model.to(device)

for epoch in range(num_epochs):
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
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

# Evaluate the model on the test set
test_loss = 0
vit_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
       

