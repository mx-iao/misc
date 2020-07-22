## Current experience

### Problem statement
Today's experience for using ScriptRunConfig requires the user to specify much of the required information by accessing the RunConfiguration property of a ScriptRunConfig object themselves. In addition, most of the PipelineSteps take in a RunConfiguration object and not a ScriptRunConfig object, creating a disconnect between standalone runs and Pipeline runs. Finally, the presence of having both ScriptRunConfig and Estimator options for configuring runs (both of which have their cons) has been a neverending source of confusion to customers.

### Goal
Until we are ready to ship the long-term work for Components in vNext, we will do interim work to improve the existing experience. We will dedupe ScriptRunConfig and Estimators by improving the ScriptRunConfig creation experience. We should be able to do so without introducing any breaking changes. Recommended paths for configuring training jobs should not require user to use RunConfiguration themselves. In addition, we will make it easier and more streamlined to configure and submit jobs from the CLI, and move from standalone runs to Pipeline runs.

### Example:
```python
src = ScriptRunConfig(source_directory=project_folder, 
                      script='train.py', 
                      arguments=['--num_epochs': '30'], 
                      run_config=RunConfiguration(framework='python', communicator='Mpi'))
                                
pt_env = Environment.get(ws, name='AzureML-PyTorch-1-3-GPU'))

src.run_config.environment = pt_env
src.run_config.node_count = 2
src.run_config.target = compute_target
src.run_config.data = Data(data_location = DataLocation(dataset=dataset), 
                           mechanism='direct', 
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



## Proposed experience
### To do
- Improve [ScriptRunConfig](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py) constructor
- Add support for specifying ScriptRunConfig from YAML, e.g. `ScriptRunConfig.from_yaml(file_name='run_config.yml')`
- Update run submission CLI (`az ml run submit-script`) to support ScriptRunConfig YAML
- Update PythonScriptStep to take in a `ScriptRunConfig` object directly instead of a `RunConfiguration` object
- `RunConfiguration` class should become an implementation detail and removed from docs/tutorials (as much as possible)
- HyperDriveConfig can already take a ScriptRunConfig object
- Make ScriptRunConfig (and PythonScriptStep) the recommended run submission path (over Estimator and EstimatorStep)
- Add robust validation (either at ScripRunConfig construction time or job submission time) to validate the combination of arguments specified by user.
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


### Example:
```python
pt_env = Environment.get(ws, name='AzureML-PyTorch-1-3-GPU'))
src = ScriptRunConfig(source_directory=project_folder,
                        compute_target=compute_target,
                        entry_script='train.py',
                        arguments=['--num_epochs': '30'],
                        node_count=2,
                        communicator=MPI(process_count_per_node=4),
                        environment=pt_env,
                        inputs=[dataset.as_named_input('cifar10')])

# as an individual run
run = experiment.submit(src)

# as a pipeline run
train_step = PythonScriptStep(name='train',
                              script_run_config=src,
                              inputs=[dataset.as_named_input('cifar10')])
                              
pipeline = Pipeline(ws, steps=[train_step])
pipeline_run = experiment.submit(pipeline)
```
