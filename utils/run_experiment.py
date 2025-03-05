import os

import torch
import wandb
import yaml
from peft import PromptLearningConfig

from SWAG import SWAG
import numpy
import random

from loraxs import find_and_initialize
from train import train_swag
from utils.peft_utils import create_peft_model, get_peft_config
from utils.config_utils import set_save_path
from data import (
    load_glue_data,
    load_mcqa_data,
    MCQA_task_to_context_keys,
    GLUE_task_to_keys,
)
from accelerate import Accelerator, DeepSpeedPlugin


def run_experiment(config):
    # deepspeed and accelerate initialization
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=config.experiment.gradient_accumulation_steps,
    )
    accelerator = Accelerator(
        split_batches=True, log_with="wandb", deepspeed_plugin=deepspeed_plugin
    )  # mixed_precision='fp16')

    wandb_group = config.experiment.wandb_group
    model_name = config.model.model_name
    task = config.experiment.task

    if config.experiment.use_loraxs:
        config.model.target_modules.extend(
            ["attention.output.dense", "output.dense"])

    active_tags = [
        model_name,
        task,
        "loraxs" if config.experiment.use_loraxs else "lora",
    ]

    accelerator.init_trackers(
        project_name=config.experiment.wandb_project,
        init_kwargs={
            "wandb": {
                "entity": config.experiment.wandb_entity,
                "group": wandb_group,
                "tags": active_tags,
            }
        },
    )

    set_save_path(config, accelerator)

    seed = config.experiment.seed
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    batch_size = config.experiment.batch_size
    task = config.experiment.task

    print("offline loading = ", config.experiment.offline)

    ood_dataloader = None

    # Load data
    if task in GLUE_task_to_keys:
        tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes = (
            load_glue_data(
                config,
                task,
                accelerator,
                batch_size=batch_size,
                subtask=config.experiment.subtask,
                offline=config.experiment.offline,
                data_path=config.experiment.data_path,
            )
        )
    elif task in MCQA_task_to_context_keys:
        tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes = (
            load_mcqa_data(
                config,
                task,
                accelerator,
                batch_size=batch_size,
                subtask=config.experiment.subtask,
                offline=config.experiment.offline,
                data_path=config.experiment.data_path,
            )
        )
    else:
        raise Exception("Only GLUE tasks and MCQA tasks implemented")

    if config.experiment.ood_task != "":
        if task in GLUE_task_to_keys:
            _, _, _, ood_dataloader, _ = load_glue_data(
                config,
                config.experiment.ood_task,
                accelerator,
                batch_size=config.experiment.ood_batch_size,
                subtask=config.experiment.ood_subtask,
                offline=config.experiment.offline,
                data_path=config.experiment.data_path,
            )
        elif task in MCQA_task_to_context_keys:
            _, _, _, ood_dataloader, _ = load_mcqa_data(
                config,
                config.experiment.ood_task,
                accelerator,
                batch_size=config.experiment.ood_batch_size,
                subtask=config.experiment.ood_subtask,
                offline=config.experiment.offline,
                data_path=config.experiment.data_path,
            )
        else:
            raise Exception("Only GLUE tasks and MCQA tasks implemented")

    peft_config = get_peft_config(config, accelerator=accelerator)

    accelerator.print(f"Method: {config.method.method_name}")

    # create peft model
    print("RANK ", peft_config.r)
    model = create_peft_model(
        config, peft_config, num_classes=num_classes, tokenizer=tokenizer
    )

    if config.experiment.use_loraxs:
        print("Creating LORA-XS")

        adapter_name = "default"
        peft_config_dict = {}
        if not isinstance(peft_config, PromptLearningConfig):
            peft_config_dict[adapter_name] = peft_config

        with open("conf/reconstruct_config.yaml", "r") as stream:
            reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
        reconstr_type = reconstr_config["reconstruction_type"]
        reconstr_config[reconstr_type]["rank"] = peft_config_dict[adapter_name].r
        print("XS-RANK ", peft_config_dict[adapter_name].r)

        find_and_initialize(
            model,
            peft_config_dict,
            adapter_name=adapter_name,
            reconstr_type=reconstr_type,
            reconstruct_config=reconstr_config,
            writer=None,
        )
        model.print_trainable_parameters()

        if config.experiment.task in ["mrpc", "rte", "stsb"]:
            if config.experiment.mnli_model_path is None:
                raise ValueError(
                    f"Path to MNLI model is not provided for {config.experiment.task} task."
                )
            elif peft_config.r in [4, 8, 12, 16, 20, 25]:
                from safetensors import safe_open

                state_dict_mnli = {}
                with safe_open(
                    os.path.join(
                        config.experiment.mnli_model_path, "adapter_model.safetensors"
                    ),
                    framework="pt",
                    device="cpu",
                ) as f:
                    for key in f.keys():
                        state_dict_mnli[key] = f.get_tensor(key)
                renamed_state_dict = {
                    k.replace("lora_A", "lora_A.default")
                    .replace("lora_B", "lora_B.default")
                    .replace("_lora_latent", ".default_lora_latent"): v
                    for (k, v) in state_dict_mnli.items()
                    if "classifier.out_proj" not in k
                }

                model.load_state_dict(renamed_state_dict, strict=False)
                print("Loaded pretrained")

        model.print_trainable_parameters()

    for param in model.parameters():
        param.data = param.data.contiguous()

    # SWAG
    if config.method.method_name == "swag":
        swag_model = SWAG(
            model,
            no_cov_mat=not config.method.swag_cov_mat,
            max_num_models=config.method.swag_max_num_models,
            modules_to_swag=config.method.modules_to_swag,
        )
        swag_model.train()

        wandb.run.tags = list(wandb.run.tags) + [f"{swag_model.lora_params}K"]

        train_swag(
            model,
            swag_model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            config,
            accelerator,
            tokenizer,
            num_classes=num_classes,
            peft_config=peft_config,
            ood_dataloader=ood_dataloader,
        )

    else:
        raise Exception("Method not implemented")

    accelerator.end_training()
