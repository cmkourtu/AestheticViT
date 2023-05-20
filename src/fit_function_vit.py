import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

role = get_execution_role()
session = sagemaker.Session()

image = '397439799241.dkr.ecr.us-east-2.amazonaws.com/aestheticvit:latest' # The ECR URI of your Docker container

estimator = Estimator(image_uri=image,
                      role=role,
                      instance_count=1,
                      instance_type='ml.m5.xlarge',  # Don't forget to complete the instance type
                      hyperparameters={
                          'epochs': 10,
                          'lr': 0.01,
                          'batch_size': 64,
                      })

s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://kourtutest/train', content_type='application/x-image')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://kourtutest/val', content_type='application/x-image')

estimator.fit({'train': s3_input_train, 'validation': s3_input_validation})