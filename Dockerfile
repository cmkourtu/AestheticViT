# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.10.0-cuda11.1-cudnn8-runtime

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src/ /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Run sagemaker_train_vit.py when the container launches
ENTRYPOINT ["python", "/app/sagemaker_train_vit.py"]
