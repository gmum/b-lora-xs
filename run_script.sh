#!/bin/bash

export RANK=0
export LOCAL_RANK=0
export MASTER_PORT=9355
export MASTER_ADDR="localhost"
export WORLD_SIZE=1

# Define the different values for experiment.lora_r, method.swag_start, and experiment.task
lora_r_values=(2 8 16 25)  # Example values for experiment.lora_r
swag_start_values=(25 50 75 100)  # Example values for method.swag_start
task_values=("cola" "sst2")  # Updated task values
seed_values=(0 1 2 3 4) # Seeds

# Loop through each combination of the values
for lora_r in "${lora_r_values[@]}"; do
  for swag_start in "${swag_start_values[@]}"; do
    for task in "${task_values[@]}"; do
      for seed in "${seed_values[@]}"; do
        mnli_model_path="./model_checkpoints/RoBERTa-large/MNLI/rank_${lora_r}"

        # Run the experiment with the current combination of parameters
        accelerate launch launch_exp_hydra.py \
          experiment.task=$task \
          method.force_save=-1 \
          method.swag_anneal_epochs=5 \
          method.swag_max_num_models=10 \
          method.swag_c_epochs=1 \
          method.swag_learning_rate=1e-3 \
          experiment.learning_rate=1e-3 \
          experiment.cls_learning_rate=5e-3 \
          experiment.num_epochs=100 \
          experiment.batch_size=32 \
          experiment.eval_upon_save=True \
          experiment.use_loraxs=True \
          experiment.lora_r=$lora_r \
          experiment.seed=$seed \
          method.swag_start=$swag_start \
          experiment.mnli_model_path=$mnli_model_path
      done
    done
  done
done