import os
import random
import shutil
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Set the paths to the AVA dataset and the AVA txt file
ava_dataset_path = 'path/to/AVA_dataset_folder'
ava_txt_file = 'path/to/AVA.txt'

# Read the AVA txt file
ava_data = pd.read_csv(ava_txt_file, delimiter=' ', header=None)

# Extract the image IDs and aesthetic scores
image_ids = ava_data.iloc[:, 1].values
aesthetic_scores = ava_data.iloc[:, 2:12].values.sum(axis=1)

# Split the dataset into train, validation, and test sets
train_image_ids, test_image_ids, train_scores, test_scores = train_test_split(
    image_ids, aesthetic_scores, test_size=0.2, random_state=42
)
train_image_ids, val_image_ids, train_scores, val_scores = train_test_split(
    train_image_ids, train_scores, test_size=0.25, random_state=42
)

# Create directories for the preprocessed data
os.makedirs('AVA_preprocessed/train', exist_ok=True)
os.makedirs('AVA_preprocessed/val', exist_ok=True)
os.makedirs('AVA_preprocessed/test', exist_ok=True)

# Function to resize and save images
def preprocess_and_save_images(image_ids, scores, dataset_path, output_path):
    for image_id, score in zip(image_ids, scores):
        image_path = os.path.join(dataset_path, f'{image_id}.jpg')
        output_image_path = os.path.join(output_path, f'{image_id}.jpg')
        
        # Resize the image
        image = Image.open(image_path)
        image = image.resize((224, 224), Image.ANTIALIAS)
        
        # Save the preprocessed image
        image.save(output_image_path)

        # Store the image aesthetic score
        with open(os.path.join(output_path, 'scores.txt'), 'a') as f:
            f.write(f'{image_id} {score}\n')

# Preprocess and save the train, validation, and test images
preprocess_and_save_images(train_image_ids, train_scores, ava_dataset_path, 'AVA_preprocessed/train')
preprocess_and_save_images(val_image_ids, val_scores, ava_dataset_path, 'AVA_preprocessed/val')
preprocess_and_save_images(test_image_ids, test_scores, ava_dataset_path, 'AVA_preprocessed/test')
