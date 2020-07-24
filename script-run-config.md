## Current experience

### Problem statement
Today's experience for using ScriptRunConfig requires the user to specify much of the required information by accessing the RunConfiguration property of a ScriptRunConfig object themselves. In addition, most of the PipelineSteps take in a RunConfiguration object and not a ScriptRunConfig object, creating a disconnect between standalone runs and Pipeline runs. Finally, the presence of having both ScriptRunConfig and Estimator options for configuring runs (both of which have their cons) has been a neverending source of confusion for customers (particularly now that introduction of EMS and curated environments has made Estimators basically redundant).

### Goal
Until we are ready to ship the long-term solution of Components in vNext, we will do interim work to improve the existing experience. We will dedupe ScriptRunConfig and Estimators by improving the ScriptRunConfig creation experience and replacing Estimators. We should be able to do so without introducing any breaking changes. Recommended paths for configuring training jobs should not require user to use RunConfiguration themselves. With these improvements, we can also stop updating the set of private framework images that we currently have. In addition, we will make it easier and more streamlined to configure and submit jobs from the CLI, and move from standalone runs to Pipeline runs.

### Example: current E2E experience
```python
src = ScriptRunConfig(source_directory=project_folder, 
                      script='train.py', 
                      arguments=['--learning-rate': '0.0001'], 
                      run_config=RunConfiguration(framework='python', communicator='Mpi'))
                                
pt_env = Environment.get(ws, name='AzureML-PyTorch-1.6-GPU'))

src.run_config.environment = pt_env
src.run_config.node_count = 4
src.run_config.target = gpu-cluster
src.run_config.data = Data(data_location = DataLocation(dataset=dataset), 
                           mechanism='mount', 
                           environment_variable_name='cifar10')

# as an individual run
run = experiment.submit(src)

# as a pipeline run
train_step = PythonScriptStep(script_name=src.script,
                              name='train',
                              arguments=src.arguments,
                              compute_target=src.run_config.target,
                              runconfig=src.run_config,
                              inputs=[dataset.as_named_input('cifar10')],
                              source_directory=src.source_directory)
                              
pipeline = Pipeline(ws, steps=[train_step])
pipeline_run = experiment.submit(pipeline)
```

### Example: current TF experience
```python
src = ScriptRunConfig(source_directory=project_folder, 
                      script='train.py', 
                      arguments=['--num-epochs': '1000'], 
                      run_config=RunConfiguration(framework='Tensorflow', communicator='ParameterServer'))
                                
tf_env = Environment.get(ws, name='AzureML-TensorFlow-1.14-GPU'))

tensorflow_configuration = TensorFlowConfiguration()	
tensorflow_configuration.worker_count = 2
tensorflow_configuration.parameter_server_count = 2

src.run_config.environment = tf_env
src.run_config.node_count = 4
src.run_config.target = gpu-cluster
src.run_config.tensorflow = tensorflow_configuration

# as an individual run
run = experiment.submit(src)
```

## Proposed experience
### To do
- Improve [ScriptRunConfig](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py) constructor
- Add support for specifying ScriptRunConfig from YAML, e.g. `ScriptRunConfig.from_yaml(file_name='run_config.yml')`
- Update run submission CLI (`az ml run submit-script`) to support ScriptRunConfig YAML
- `RunConfiguration` class should become an implementation detail and removed from docs/tutorials for as many scenarios as possible
- Updates to `MpiConfiguration`, `TensorFlowConfiguration`, `PyTorchConfiguration` (add)
- HyperDriveConfig can already take a ScriptRunConfig object
- Add robust validation for the combination of arguments specified by user
- Update PythonScriptStep to take in a `ScriptRunConfig` object directly instead of a `RunConfiguration` object
- Make ScriptRunConfig (and PythonScriptStep) the recommended run submission path (over Estimator and EstimatorStep)
- Update documentation, notebooks, TSGs

### ScriptRunConfig constructor
```python
class ScriptRunConfig(ABC):
	""" ScriptModule is a configuration for a script-based job
	:raises:
	:rtype:
	"""
	def __init__(self,
                  source_directory,
                  command,
                  compute_target=None,
                  inputs=None,
                  environment=None, 
                  job_config=None, 
                  node_count=1,
                  source_directory_data_store=None,
                  resume_from=None,
                  max_run_duration_seconds=3600*24*30):
		pass
```

#### Arguments
| name | type | description |
| ---- | ---- | ----------- |
| source_directory | str | A local directory containing experiment configuration and code files needed for a training job. |
| command | str or list(str) | The command to run on the compute target including the command-line arguments to pass to the in.` |
| compute_target | str or ComputeTarget | The compute target where training will happen. This can either be a ComputeTarget object or the string "local". |
| inputs | list(DataReference or DatasetConsumptionConfig) | List of DataReference or DatasetConsumptionConfig for datasets to use as inputs for run. |
| environment | Environment | The environment to use for the run. If no environment is specified, `DEFAULT_CPU_IMAGE` will be used as the Docker image for the run. |
| job_config | TensorFlowConfiguration, MpiConfiguration, PyTorchConfiguration, ParallelTaskConfiguration | For jobs that require additional job-specific configurations, e.g. distributed training jobs. |
| node_count | int | The number of nodes to use for the job. |
| source_directory_data_store | Datastore | The backing datastore for the project share. |
| resume_from | DataPath | The DataPath containing the checkpoint or model files from which to resume the experiment. |
| max_run_duration_seconds | int | The maximum time allowed for the run. The system will attempt to automatically cancel the run if it took longer than this value. |


### Example: Scikit-learn
```python
sklearn_env = Environment.get(ws, name='AzureML-Scikit-learn-0.23.1')
sklearn_runconfig = ScriptRunConfig(source_directory=project_folder,
				    command=['python', 'train.py', '--data-folder', dataset.as_download()],
				    compute_target=cpu-cluster,
				    environment=sklearn_env)

run = experiment.submit(sklearn_runconfig)
```

### Example: TensorFlow
```python
tf_env = Environment.get(ws, name='AzureML-TensorFlow-2.1-GPU')
tf_runconfig = ScriptRunConfig(source_directory=project_folder,
			       command=['python', 'train.py', '--num-epochs', '1000'],
			       compute_target=gpu-cluster,
			       environment=tf_env,
			       job_config=TensorflowConfiguration(worker_count=4),
			       node_count=4)

run = experiment.submit(tf_runconfig)
```

### Example: PyTorch
```python
pt_env = Environment.get(ws, name='AzureML-PyTorch-1.6-GPU')
pt_runconfig = ScriptRunConfig(source_directory=project_folder,
			       command=['python', 'train.py', '--dist-backend', 'nccl', '--dist-url', '$AZ_BATCHAI_PYTORCH_INIT_METHOD', '--rank', '$AZ_BATCHAI_TASK_INDEX'],
			       compute_target=gpu-cluster,
			       environment=pt_env,
			       job_config=PyTorchConfiguration(communication_backend='NCCL'),
			       node_count=4)

run = experiment.submit(pt_runconfig)
```

### Example: MPI
```python
pt_env = Environment.get(ws, name='AzureML-PyTorch-1.6-GPU')
horovod_runconfig = ScriptRunConfig(source_directory=project_folder,
				    command=['python', 'train.py', '--learning-rate', 0.001],
				    compute_target=gpu-cluster,
				    environment=pt_env,
				    job_config=MpiConfiguration(process_count_per_node=2),
				    node_count=4)

run = experiment.submit(horovod_runconfig)
```

### Example: HyperDrive
```python
param_sampling = RandomParameterSampling( {
        'learning_rate': uniform(0.0005, 0.005)
    }
)

early_termination_policy = BanditPolicy(slack_factor=0.15, evaluation_interval=1, delay_evaluation=10)

hd_config = HyperDriveConfig(run_config=horovod_runconfig,
			     hyperparameter_sampling=param_sampling, 
			     policy=early_termination_policy,
			     primary_metric_name='best_val_acc',
			     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
			     max_total_runs=20,
			     max_concurrent_runs=4)
			     
hd_run = experiment.submit(hd_config)
```

### Example: Pipeline
```python
pt_env = Environment.get(ws, name='AzureML-PyTorch-1.6-GPU'))
src = ScriptRunConfig(source_directory=project_folder,
		      command=['python', 'train.py', '--learning-rate', 0.001],
                      compute_target=gpu-cluster,
		      node_count=4,
		      environment=pt_env,
                      job_config=MpiConfiguration(process_count_per_node=2),
		      inputs=[dataset.as_named_input('cifar10')])

# as an individual run
run = experiment.submit(src)

# as a pipeline run
train_step = PythonScriptStep(name='train',
                              script_run_config=src)
                              
pipeline = Pipeline(ws, steps=[train_step])
pipeline_run = experiment.submit(pipeline)
```

### Example: Reinforcement Learning (based on existing preview)
```python
worker_config = WorkerConfiguration(compute_target=worker-cluster,
				    node_count=4,
				    environment=worker_env)
				    
rl_config = RLConfiguration(worker_configuration=worker_config, 
			    rl_framework=Ray(version='0.8.3'),
			    simulator_config=None)

rl_runconfig = ScriptRunConfig(source_directory=project_folder,
			       command=['python', 'pong_rllib.py', '--run', 'IMPALA', '--env', '$PongNoFrameskip-v4', '--config', '\'{"num_gpus": 1, "num_workers": 13}\'', '--stop', '\'{"episode_reward_mean": 18, "time_total_s": 3600}\''],
			       compute_target=head-cluster,
			       environment=gpu_pong_env,
			       job_config=rl_config)

run = experiment.submit(rl_runconfig)
```
