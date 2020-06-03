```
src = ScriptRunConfig(source_directory=".", 
                      script='train.py', 
                      arguments=['--num_epochs': '30'], 
                      run_config=RunConfiguration(framework='python', communicator='ParameterServer'))

src.run_config.environment = Environment.get(ws, name='AzureML-TensorFlow-2-0-GPU'))
src.run_config.node_count = 2
src.run_config.target = compute_target
```
