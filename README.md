# AestheticViT

AestheticViT is a project for training a Vision Transformer (ViT) model to predict the aesthetic scores of images using the AVA dataset.

## Prerequisites

- Python 3.7 or later
- PyTorch 1.9 or later
- torchvision 0.10 or later
- Pillow 8.0 or later
- pandas 1.0 or later
- scikit-learn 0.24 or later

You can install the required packages using the following command:

```bash
pip install torch torchvision Pillow pandas scikit-learn
```

## Dataset

The project uses the AVA dataset ,swhich can be downloaded from the official website or a trusted source. The dataset should be placed in the data/AVA_dataset folder with the following structure:

```objectivec
data/AVA_dataset/
    ├── images/
    └── AVA.txt
```
## Usage

1. Preprocess the AVA dataset by running the preprocess_ava.py script. This script resizes the images to 224x224 pixels, splits the dataset into train, validation, and test sets, and stores the preprocessed data in the data/AVA_preprocessed folder.

```bash
python src/preprocess_ava.py
```
2. Train the ViT model on the preprocessed dataset using the train_vit.py script. This script trains the model for a specified number of epochs and saves the best model based on validation loss.

```bash
python src/train_vit.py
```

## License

This project is released under the MIT License. See the LICENSE file for more details.
