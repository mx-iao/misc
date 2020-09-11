## Example: CNN Daily Mail
https://github.com/microsoft/Turing-NLR/blob/main/notebooks/CNNDailyMail.ipynb

### With 1.13.0
```python
source_directory = "../examples/cnndailymail_finetune"

workdir = "/turing/workdir"
train_file_path = "final_datasets/cnndm/cnndm.train.uncased_tokenized.json"
model_checkpoint_path = "tnlrv3-base.pt"
output_data_path = "amlturing/nlg_50epoch_128batch_final/"

mpi = MpiConfiguration()
mpi.process_count_per_node = 4

args = ["--train_file", input_ds.path(train_file_path).as_download(), 
        "--upload_dir", default_ds.path(output_data_path).as_mount(), 
        "--output_dir", workdir,
        "--model_name_or_path", input_ds.path(model_checkpoint_path).as_download(),
        "--do_lower_case",
        "--fp16",
        "--fp16_opt_level", "O2",
        "--tokenizer_name", "./tnlr/tokenizer/tnlr-base-uncased-vocab.txt",
        "--config_name", "./tnlr/config/tnlr-base-uncased-config.json",
        "--max_source_seq_length", 608,
        "--max_target_seq_length", 160,
        "--per_gpu_train_batch_size", 8,
        "--gradient_accumulation_steps", 1,
        "--learning_rate", 7e-5,
        "--num_warmup_steps", 1000,
        "--num_training_epochs", 45,
        "--writers": "aml",  # use 'aml-tensorboard' to also put tensforboard events including weights histograms
        "--save_steps": 5000,
        "--target_mask_prob": 0.8]

finetune_src = ScriptRunConfig(source_directory=source_directory,
                               script="run_seq2seq.py",
                               arguments=args,
                               compute_target=gpu_compute_target,
                               environment=myenv,
                               distributed_job_config=mpi)
                               
finetune.run_config.node_count = 4
```

### Final
```python
source_directory = "../examples/cnndailymail_finetune"

workdir = "/turing/workdir"
train_file_path = "final_datasets/cnndm/cnndm.train.uncased_tokenized.json"
model_checkpoint_path = "tnlrv3-base.pt"
output_data_path = "amlturing/nlg_50epoch_128batch_final/"

mpi = MpiConfiguration(process_count_per_node=4, node_count=4)

args = ["--train_file", input_ds.path(train_file_path).as_download(), 
        "--upload_dir", default_ds.path(output_data_path).as_mount(), 
        "--output_dir", workdir,
        "--model_name_or_path", input_ds.path(model_checkpoint_path).as_download(),
        "--do_lower_case",
        "--fp16",
        "--fp16_opt_level", "O2",
        "--tokenizer_name", "./tnlr/tokenizer/tnlr-base-uncased-vocab.txt",
        "--config_name", "./tnlr/config/tnlr-base-uncased-config.json",
        "--max_source_seq_length", 608,
        "--max_target_seq_length", 160,
        "--per_gpu_train_batch_size", 8,
        "--gradient_accumulation_steps", 1,
        "--learning_rate", 7e-5,
        "--num_warmup_steps", 1000,
        "--num_training_epochs", 45,
        "--writers": "aml",  # use 'aml-tensorboard' to also put tensforboard events including weights histograms
        "--save_steps": 5000,
        "--target_mask_prob": 0.8]

finetune_src = ScriptRunConfig(source_directory=source_directory,
                               script="run_seq2seq.py",
                               arguments=args,
                               compute_target=gpu_compute_target,
                               environment=myenv,
                               distributed_job_config=mpi)
```
