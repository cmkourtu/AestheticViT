import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

role = get_execution_role()
session = sagemaker.Session()

image = '397439799241.dkr.ecr.us-east-2.amazonaws.com/aestheticvit:latest'

estimator = Estimator(image_uri=image,
                      role=role,
                      instance_count=1,
                      instance_type='ml.g4dn.2xlarge',
                      max_run=86400,
                      use_spot_instances=True,  # use spot instances
                      max_wait=200000,  # Maximum waiting time
                      checkpoint_s3_uri='s3://kourtutest/checkpoints/',  # S3 Uri for storing checkpoints
                      hyperparameters={
                          'epochs': 10,
                          'lr': 0.0006994586775569564,
                          'batch_size': 55,
                      })

# Define the hyperparameter ranges
hyperparameter_ranges = {
    'lr': ContinuousParameter(0.01, 32),
    'batch_size': IntegerParameter(32, 64),
    'epochs': IntegerParameter(5, 10)
}

# Define the objective metric
objective_metric_name = 'Validation Loss'
objective_type = 'Minimize'
metric_definitions = [{'Name': 'Validation Loss',
                       'Regex': 'Validation Loss: ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'}]

# Create a HyperparameterTuner
tuner = HyperparameterTuner(estimator,
                            objective_metric_name,
                            hyperparameter_ranges,
                            metric_definitions,
                            max_jobs=9,
                            max_parallel_jobs=9,
                            objective_type=objective_type)

s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://kourtutest/train', content_type='application/x-image')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://kourtutest/val', content_type='application/x-image')
s3_input_test = sagemaker.inputs.TrainingInput(s3_data='s3://kourtutest/test', content_type='application/x-image')

# Start the tuning job
tuner.fit({'train': s3_input_train, 'val': s3_input_validation, 'test': s3_input_test})
