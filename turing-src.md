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

## Example: Glue-SST
https://github.com/microsoft/Turing-NLR/blob/main/notebooks/GLUE-SST.ipynb

### With 1.13.0
```python
mpi = MpiConfiguration()
mpi.process_count_per_node = 4

model_checkpoint_path = "tnlrv3-base.pt"
dataset_path = "glue_data/SST-2/"
output_path = "outputs/glue-outputs/"

args = ["--model_name_or_path", ds.path(model_checkpoint_path).as_download(),
        "--task_name", "sst-2",
        "--tokenizer_name", "./tnlr/tokenizer/tnlr-base-uncased-vocab.txt",
        "--config_name", "./tnlr/config/tnlr-base-uncased-config.json",
        "--do-train",
        "--do-lower-case",
        "--evaluate_during_training",
        "--data_dir", default_ds.path(dataset_path).as_mount(),
        "--output_dir", default_ds.path(output_path).as_mount(),
        "--max_seq_length", 128,
        "--per_gpu_train_batch_size", 32,
        "--learning_rate", 7e-6,
        "--num_train_epochs", 15.0,
        "--weight_decay", 0.01,
        "--fp16",
        "--fp16_opt_level", "O2",
        "--overwrite_output_dir",
        "--do_eval"]

finetune_src = ScriptRunConfig(source_directory=source_directory,
                               script="run_classifier.py",
                               arguments=args,
                               compute_target=gpu_compute_target,
                               environment=myenv,
                               distributed_job_config=mpi)

finetune_src.run_config.node_count = 1
```

### Final
```python
mpi = MpiConfiguration(process_count_per_node=4, node_count=1)

model_checkpoint_path = "tnlrv3-base.pt"
dataset_path = "glue_data/SST-2/"
output_path = "outputs/glue-outputs/"

args = ["--model_name_or_path", ds.path(model_checkpoint_path).as_download(),
        "--task_name", "sst-2",
        "--tokenizer_name", "./tnlr/tokenizer/tnlr-base-uncased-vocab.txt",
        "--config_name", "./tnlr/config/tnlr-base-uncased-config.json",
        "--do-train",
        "--do-lower-case",
        "--evaluate_during_training",
        "--data_dir", default_ds.path(dataset_path).as_mount(),
        "--output_dir", default_ds.path(output_path).as_mount(),
        "--max_seq_length", 128,
        "--per_gpu_train_batch_size", 32,
        "--learning_rate", 7e-6,
        "--num_train_epochs", 15.0,
        "--weight_decay", 0.01,
        "--fp16",
        "--fp16_opt_level", "O2",
        "--overwrite_output_dir",
        "--do_eval"]

finetune_src = ScriptRunConfig(source_directory=source_directory,
                               script="run_classifier.py",
                               arguments=args,
                               compute_target=gpu_compute_target,
                               environment=myenv,
                               distributed_job_config=mpi)
```

## Example: SuggestedReplies
https://github.com/microsoft/Turing-NLR/blob/main/notebooks/SuggestedReplies.ipynb

### With 1.13.0
```python
mpi = MpiConfiguration()
mpi.process_count_per_node = 4

source_directory = "../examples/suggestedreplies_finetune"

input_data = input_ds.path("final_datasets").as_download(
    path_on_compute="/turing"
)
model_data = input_ds.path("tnlrv3-base.pt").as_download(
    path_on_compute="/turing"
)
output_data = default_ds.path("amlturing/SR").as_mount()

args = ["--cf": "./configs/tnlr_base_config_ort.json",
        "--model_checkpoint_dir", output_data,
        "--input_data_dir", input_data,
        "--model_data_dir", model_data,
        "--fp16",
        "--do_train",
        "--do_valid",
        "--do_test",
        "--do_lower_case",
        "--valid_interval", 2000,
        "--use_basic_logger",
        "--writers", "aml"]

src = ScriptRunConfig(source_directory=source_directory,
                      script="train_ort.py",
                      arguments=args,
                      compute_target=gpu_compute_target,
                      environment=myenv,
                      distributed_job_config=mpi)

src.run_config.node_count = 4
```

Note: add args for `--input_data_dir` and `--model_data_dir` and parse in training script.

### Final
```python
mpi = MpiConfiguration(process_count_per_node=4, node_count=4)

source_directory = "../examples/suggestedreplies_finetune"

input_data = input_ds.path("final_datasets").as_download(
    path_on_compute="/turing"
)
model_data = input_ds.path("tnlrv3-base.pt").as_download(
    path_on_compute="/turing"
)
output_data = default_ds.path("amlturing/SR").as_mount()

args = ["--cf": "./configs/tnlr_base_config_ort.json",
        "--model_checkpoint_dir", output_data,
        "--input_data_dir", input_data,
        "--model_data_dir", model_data,
        "--fp16",
        "--do_train",
        "--do_valid",
        "--do_test",
        "--do_lower_case",
        "--valid_interval", 2000,
        "--use_basic_logger",
        "--writers", "aml"]

src = ScriptRunConfig(source_directory=source_directory,
                      script="train_ort.py",
                      arguments=args,
                      compute_target=gpu_compute_target,
                      environment=myenv,
                      distributed_job_config=mpi)
```
