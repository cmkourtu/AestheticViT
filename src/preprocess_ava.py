import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Set the paths to the AVA dataset and the AVA txt file
ava_dataset_path = 'data/AVA_dataset/images'
ava_txt_file = 'data/AVA_dataset/AVA.txt'

# Read the AVA txt file
ava_data = pd.read_csv(ava_txt_file, delimiter=' ', header=None)

# Extract the image IDs and aesthetic scores
image_ids = ava_data.iloc[:, 1].values
scores = ava_data.iloc[:, 2:12].values

# Define the weights (score values from 1 to 10)
weights = np.arange(1, 11)

# Compute the weighted average of the scores
weighted_aesthetic_scores = np.average(scores, axis=1, weights=weights)

# Split the dataset into train, validation, and test sets
train_image_ids, test_image_ids, train_scores, test_scores = train_test_split(
    image_ids, weighted_aesthetic_scores, test_size=0.2, random_state=42
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
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"File {image_path} not found.")
            continue

        # Resize the image
        try:
            image = Image.open(image_path)
            image = image.resize((224, 224), Image.ANTIALIAS)
            
            # Save the preprocessed image
            image.save(output_image_path)

            # Store the image aesthetic score
            with open(os.path.join(output_path, 'scores.txt'), 'a') as f:
                f.write(f'{image_id} {score}\n')
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Preprocess and save the train, validation, and test images
preprocess_and_save_images(train_image_ids, train_scores, ava_dataset_path, 'AVA_preprocessed/train')
preprocess_and_save_images(val_image_ids, val_scores, ava_dataset_path, 'AVA_preprocessed/val')
preprocess_and_save_images(test_image_ids, test_scores, ava_dataset_path, 'AVA_preprocessed/test')
