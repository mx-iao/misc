```
src = ScriptRunConfig(source_directory=".", 
                      script='train.py', 
                      arguments=['--num-epochs': '30'], 
                      run_config=RunConfiguration(framework='TensorFlow', communicator='ParameterServer'))

src.run_config.environment = Environment.get(ws, name='AzureML-TensorFlow-2.0-GPU'))
src.run_config.node_count = 2
src.run_config.target = compute_target
tensorflow_configuration = TensorFlowConfiguration()
tensorflow_configuration.worker_count = 2 #TensorFlowConfiguration() also has a parameter_server_count attribute
src.run_config.tensorflow = tensorflow_configuration

exp = Experiment(workspace=ws, name='my-experiment')
run = exp.submit(src)
```

```
tf_est= TensorFlow(source_directory=".",
                   compute_target=compute_target,
                   script_params={"--num-epochs": 30},
                   entry_script='train.py',
                   node_count=2,
                   distributed_training=ParameterServer(worker_count=2),
                   use_gpu=True,
                   framework_version='2.0'
                   )

run = exp.submit(tf_est)
```
