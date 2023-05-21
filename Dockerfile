# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory to /app
WORKDIR /app

# Install gcc and other dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY src/ /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the training script to the location expected by SageMaker
COPY src/sagemaker_train_vit.py /opt/ml/code/sagemaker_train_vit.py


# Set the training script as the entrypoint
ENV SAGEMAKER_PROGRAM sagemaker_train_vit.py