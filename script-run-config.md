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
**P0**
- Improve [ScriptRunConfig](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py) constructor
- Update `MpiConfiguration`, `TensorFlowConfiguration`, add `PyTorchConfiguration`
- Add validation for the combination of arguments specified by user
- Update documentation, notebooks, TSGs and make ScriptRunConfig (and PythonScriptStep) the recommended run submission path over Estimator (and EstimatorStep) -- `RunConfiguration` class should become an implementation detail and removed from docs/tutorials for as many scenarios as possible

**P1**
- Will continue to use RunConfiguration for serialization/deserialization for SDK v1 -- validate that this continues to work for run submission CLI.

**P2**
- Update PythonScriptStep (and other PipelineSteps?) to take in a `ScriptRunConfig` object directly instead of a `RunConfiguration` object and not require user to provide the flat list of parameters to the Step constructors that are redundant with the ScriptRunConfig object.
- Once Execution Service/Pipelines team adds support for generic command, add `command` parameter to ScriptRunConfig (e.g. `command=['python', 'train.py', '--arg', 'argvalue']`). This also aligns with vNext.


### ScriptRunConfig constructor
```python
class ScriptRunConfig(ABC):
	""" ScriptModule is a configuration for a script-based job
	:raises:
	:rtype:
	"""
	def __init__(self,
                  source_directory,
                  script,
		  arguments=None,
                  compute_target=None,
                  environment=None, 
                  job_config=None, 
                  node_count=1,
                  resume_from=None,
                  max_run_duration_seconds=3600*24*30):
		pass
```

#### Arguments
| name | type | description |
| ---- | ---- | ----------- |
| source_directory | str | A local directory containing experiment configuration and code files needed for a training job. |
| script | str | The file path relative to the `source_directory` of the script to be run. |
| arguments | list or str | Optional command-line arguments to pass to the `script`. |
| compute_target | str or ComputeTarget | The compute target where training will happen. This can either be a ComputeTarget object or the string "local". |
| environment | Environment | The environment to use for the run. If no environment is specified, `DEFAULT_CPU_IMAGE` will be used as the Docker image for the run. |
| job_config | TensorFlowConfiguration, MpiConfiguration, PyTorchConfiguration, ParallelTaskConfiguration | For jobs that require additional job-specific configurations, e.g. distributed training jobs. |
| node_count | int | The number of nodes to use for the job. |
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
		      command=['python', 'train.py', '--learning-rate', 0.001, '--data', dataset.as_named_input('cifar10')],
                      compute_target=gpu-cluster,
		      node_count=4,
		      environment=pt_env,
                      job_config=MpiConfiguration(process_count_per_node=2))

# as an individual run
run = experiment.submit(src)

# as a pipeline run
train_step = PythonScriptStep(name='train',
                              script_run_config=src)
                              
pipeline = Pipeline(ws, steps=[train_step])
pipeline_run = experiment.submit(pipeline)
```

### Scenario: Reinforcement Learning
Based on discussions during 7/27 spec review: since RL uses its own service (similar to HyperDrive), it would probably make the most sense for the team to add an RLConfig (similar to the HyperDriveConfig) that gets passed to `experiment.submit()` since they are not going through ScriptRunConfig today. Depending on when RL is targeted to GA, it should either align to the v1 runconfig route, or align with Modules/Components if it coincides with the vNext release.

