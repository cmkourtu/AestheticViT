import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

role = get_execution_role()
session = sagemaker.Session()

image = '397439799241.dkr.ecr.us-east-2.amazonaws.com/aestheticvit:latest'

estimator = Estimator(image_uri=image,
                      role=role,
                      instance_count=1,
                      instance_type='ml.p3.8xlarge',  # Updated to 'p3.8xlarge'
                      max_run=86400,               # Maximum training time in seconds
                      hyperparameters={
                          'epochs': 10,
                          'lr': 0.01,
                          'batch_size': 64,
                      },
                      metric_definitions=[
                          {'Name': 'Train Loss', 'Regex': '.*"metric": "Train Loss", "value": ([0-9\\.]+),.*'},
                          {'Name': 'Validation Loss', 'Regex': '.*"metric": "Validation Loss", "value": ([0-9\\.]+),.*'}
                      ])

s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://kourtutest/train', content_type='application/x-image')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://kourtutest/val', content_type='application/x-image')
s3_input_test = sagemaker.inputs.TrainingInput(s3_data='s3://kourtutest/test', content_type='application/x-image')

estimator.fit({'train': s3_input_train, 'val': s3_input_validation, 'test': s3_input_test})