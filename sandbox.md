```
src = ScriptRunConfig(source_directory=".", 
                      script='train.py', 
                      arguments=['--num_epochs': '30'], 
                      run_config=RunConfiguration(framework='TensorFlow', communicator='ParameterServer'))

src.run_config.environment = Environment.get(ws, name='AzureML-TensorFlow-2.0-GPU'))
src.run_config.node_count = 2
src.run_config.target = compute_target
src.run_config.tensorflow = TensorFlowConfiguration(worker_count=2) #TensorFlowConfiguration() also has a parameter_server_count argument

exp = Experiment(workspace=ws, name='my-experiment')
run = exp.submit(src)
```

```

```
