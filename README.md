# Minimal Ranks, Maximum Confidence: Parameter-efficient Uncertainty Quantification for LoRA

Code repository for the paper: [Minimal Ranks, Maximum Confidence: Parameter-efficient Uncertainty Quantification for LoRA](https://arxiv.org/abs/2502.12122)

Our repository is based on [https://github.com/fortuinlab/swag-lora](https://github.com/fortuinlab/swag-lora) and 
[https://github.com/MohammadrezaBanaei/LoRA-XS](https://github.com/MohammadrezaBanaei/LoRA-XS)

## Running experiments
Experimental setup is derived from the original repository and is presented below.

We use Accelerate and Hydra to run our experiments. Please see instructions [here](./sc_venv_template/readme.md) 
for guidance in setting up the environment.

### Running training script

#### Running training using Hydra
To launch a training run locally:
```
accelerate launch launch_exp_hydra.py \
method=swag \
experiment.task=cola \
method.force_save=-1 \
method.swag_start=25 \
method.swag_anneal_epochs=5 \
method.swag_learning_rate=1e-3 \
experiment.learning_rate=1e-3 \
experiment.cls_learning_rate=5e-3 \
experiment.num_epochs=100 \
experiment.batch_size=32
```

#### Running training via job script 
In [run_script.sh](run_script.sh) is an example script used in our study.

### Notable flags/hyperparameters

Important hydra flags/hyperparameters are discussed below. See a more comprehensive description of hyperparameters in
[config.yaml](./conf/config.yaml) and [swag.yaml](./conf/method/swag.yaml). 
SWAG-specific flags are represented as `method.<var_name>` as in Hydra.



| Hyperparameter                  | Description                                                                                                                      | Possible Values                                                                                |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `model`                         | Model to use                                                                                                                     | `llama_7b_hf`, `roberta-base`, `roberta-large`                                                 |
| `method`                        | The method configuration to use                                                                                                  | Only `swag`                                                                                    |
| `method.swag_learning_rate`     | Learning rate for the SWAG training                                                                                              | Any float (e.g., `1e-4`)                                                                       |
| `method.swag_anneal_epochs`     | Number of epochs for annealing in SWAG                                                                                           | Any integer (e.g., `5`)                                                                        |
| `method.swag_anneal_strategy`   | Strategy for annealing in SWAG                                                                                                   | `constant`, `linear`, `cosine`, `cosine_hard_restarts`                                         |
| `method.swag_start`             | The epoch to start SWAG collection                                                                                               | Any integer (e.g., `8`)                                                                        |
| `method.modules_to_swag`        | Modules over which to learn the SWAG distribution (supports only LoRA layers, only layers with gradients, and all)               | `grad_only`, `lora_only`, `all`                                                                |
| `method.swag_max_num_models`    | Maximum number of models to maintain for covariance approximation in SWAG                                                        | Any integer (e.g., `5`)                                                                        |
| `method.swag_cov_mat`           | Whether to learn a covariance matrix in SWAG (if `False`, only a diagonal covariance is used when sampling)                      | `True`, `False`                                                                                |
| `method.force_save`             | Force save epoch for early stopping SWAG training; SWAG model will be saved `method.force_save` epochs after `method.swag_start` | Any integer (e.g. 5)                                                                           |
| `method.swag_sample_scale`      | Covariance scale of the learned SWAG distribution we are sampling from                                                           | Any float (e.g. 1.0)                                                                           |
| `method.swag_samples`           | Number of SWAG model samples used durung evaluation                                                                              | Any integer (e.g. 15)                                                                          |
| `experiment.task`               | The (ID) task for training/evaluation                                                                                            | `obqa`, `cqa`, `swag`, `mmlu`, `arc-e`, `arc-c`, `cola`, `mnli`, `mrpc`, (other GLUE tasks...) |
| `experiment.subtask`            | (ID) Subtask for training/evaluation                                                                                             | Subtask name (e.g. `experiment.task=mmlu`, `experiment.subtask=anatomy`)                       |
| `experiment.ood_task`           | (OOD) task for evaluation                                                                                                        | `obqa`, `cqa`, `swag`, `mmlu`, `arc-e`, `arc-c`, `cola`, `mnli`, `mrpc`, (other GLUE tasks...) |
| `experiment.ood_subtask`        | (OOD) subtask for evaluation task.                                                                                               | Subtask name (e.g. `experiment.ood_task=mmlu`, `experiment.ood_subtask=anatomy`)               |
| `experiment.ood_batch_size`     | Batch size for the OOD  task                                                                                                     | Any integer (e.g., `32`)                                                                       |
| `experiment.learning_rate`      | The learning rate for training                                                                                                   | Any float (e.g., `0.001`, `0.01`)                                                              |
| `experiment.cls_learning_rate`  | The learning rate used for classifier part                                                                                       | Any float (e.g., `0.001`, `0.01`)                                                              |
| `experiment.num_epochs`         | Total number of epochs for training                                                                                              | Any integer (e.g., `20`)                                                                       |
| `experiment.batch_size`         | Batch size for the training                                                                                                      | Any integer (e.g., `16`)                                                                       |
| `experiment.set_fraction`       | Fraction of training dataset used for learning                                                                                   | Any float between 0 and 1, inclusively                                                         |
| `experiment.overwrite`          | Whether to overwrite existing experiments                                                                                        | `True`, `False`                                                                                |
| `experiment.data_path`          | Path to the data folder for the tasks (only required if `experiment.offline=True`)                                               | `/path/to/your/data/folder`                                                                    |
| `experiment.model_path`         | Path to the model folder  (only required if `experiment.offline=True`)                                                           | `/path/to/your/models/folder`                                                                  |
| `experiment.mnli_model_path`    | Path to the pretrained LoRa-XS weights for given rank on MNLI task (only for RTE and MRPC tasks)                                 | `/path/to/your/checkpoint/folder`                                                              |
| `experiment.wandb_path`         | Path to the Weights and Biases logging folder                                                                                    | `/path/to/your/wandb/folder`                                                                   |
| `experiment.wandb_group`        | Group name for Weights and Biases tracking                                                                                       | Any group name                                                                                 |
| `experiment.wandb_entity`       | Wandb entity                                                                                                                     | Any entity name                                                                                |
| `experiment.wandb_project`      | Wandb project name                                                                                                               | Any project name                                                                               |
| `experiment.wandb_mode`         | Wandb running mode                                                                                                               | `online` or `offline`                                                                          |
| `experiment.exp_name`           | (Optional) Name of the experiment                                                                                                | Any experiment name                                                                            |

## License
For the code from [https://github.com/fortuinlab/swag-lora](https://github.com/fortuinlab/swag-lora) we retain 
the original MIT [license](SWAG_LORA_LICENSE).