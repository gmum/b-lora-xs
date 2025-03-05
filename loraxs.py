# Code taken from the LoRa-XS repository (https://github.com/MohammadrezaBanaei/LoRA-XS).

import math
import types
from typing import Tuple

import numpy as np
import peft
import torch
from peft.import_utils import is_bnb_available
from peft.utils import _get_submodules
from sklearn.decomposition import TruncatedSVD
from torch.nn import init
from tqdm import tqdm
import torch.nn.functional as F


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def get_delta_weight(self, adapter) -> torch.Tensor:
    # This function is introduced in newer PEFT versions. we modify this function instead of modifying
    # the merge function (as we did previously for version 0.4.0 of PEFT).
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    output_tensor = (
        transpose(
            weight_B @ self.default_lora_latent_mapping.weight @ weight_A,
            self.fan_in_fan_out,
        )
        * self.scaling[adapter]
    )

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor


def forward_latent(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(
            x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
        )
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(
            x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
        )

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        result += (
            self.lora_B[self.active_adapter[0]](
                self.default_lora_latent_mapping(
                    self.lora_A[self.active_adapter[0]](
                        self.lora_dropout[self.active_adapter[0]](x)
                    )
                )
            )
            * self.scaling[self.active_adapter[0]]
        )
    else:
        result = F.linear(
            x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
        )

    result = result.to(previous_dtype)

    return result


def run_svd(
    input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int
) -> Tuple[np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=rank, n_iter=n_iter,
                       random_state=random_state)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd


def get_linear_rec_svd(
    input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_matrix, svd = run_svd(input_matrix, rank, n_iter, random_state)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_


def get_replacement_module(weight, module_name, type, writer, reconstruct_config):
    cfg = reconstruct_config[type]
    if type == "svd":
        reconstructed_matrix, enc, dec = get_linear_rec_svd(
            weight.cpu().detach().numpy(),
            cfg["rank"],
            cfg["n_iter"],
            cfg["random_state"],
        )
        final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
    else:
        raise NotImplementedError(f"{type} is currently not supported.")
    return final_enc, final_dec


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(target_module, new_weight):
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def update_decoder_weights(target_module, new_weight):
    device = target_module.weight.device
    with torch.no_grad():
        target_module.weight.copy_(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def kaiming_uniform_init_lower_half(matrix: torch.tensor):
    rows, _ = matrix.size()
    init.kaiming_uniform_(matrix[math.ceil(rows / 2):, :], a=math.sqrt(5))
    return matrix


def kaiming_uniform_init(matrix: torch.tensor):
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix


def find_and_initialize(
    model, peft_config, adapter_name, reconstr_type, reconstruct_config, writer
):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    half_init_dec = reconstruct_config["half_init_dec"]
    replacement_module_random_init = reconstruct_config[
        "replacement_module_random_init"
    ]
    reconstruction_mode = reconstruct_config["reconstr_mode"]
    lora_config = peft_config[adapter_name]
    r_squared = reconstruct_config[
        "r_squared"
    ]  # whether using r*r matrix between lora_A and lora_B or not
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    assert not isinstance(lora_config.target_modules, str)
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        target_module_found = any(
            key.endswith(target_key) for target_key in lora_config.target_modules
        )
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)

            if reconstruction_mode == "separated":
                replacement_encoder_weight, replacement_decoder_weight = (
                    get_replacement_module(
                        weight=target.weight.T,
                        module_name=key,
                        type=reconstr_type,
                        writer=writer,
                        reconstruct_config=reconstruct_config,
                    )
                )

                if not isinstance(target, peft.tuners.lora.Linear):
                    raise NotImplementedError(
                        "Only initialization for peft.tuners.lora.Linear type is implemented."
                    )
                    # TODO implement for Linear8bitLt
                else:
                    if half_init_dec:
                        kaiming_uniform_init_lower_half(
                            replacement_decoder_weight)
                    if replacement_module_random_init:
                        kaiming_uniform_init(replacement_encoder_weight)
                        kaiming_uniform_init(replacement_decoder_weight)
                    replace_module_weights(
                        target.lora_B.default, replacement_decoder_weight.T
                    )
                    if r_squared:
                        target.forward = types.MethodType(
                            forward_latent, target)
                        target.get_delta_weight = types.MethodType(
                            get_delta_weight, target
                        )
                        replace_module_weights(
                            target.lora_A.default, replacement_encoder_weight.T
                        )
                        target.default_lora_latent_mapping = torch.nn.Linear(
                            lora_config.r, lora_config.r, bias=False
                        )
                        init_module_weights(
                            target.default_lora_latent_mapping, sigma=0.00001
                        )
                        target.default_lora_latent_mapping.to(
                            target.lora_A.default.weight.device
                        )

                        target.lora_A.default.weight.requires_grad = (
                            False  # only the r*r matrix will be tuned
                        )
                        target.lora_B.default.weight.requires_grad = (
                            False  # only the r*r matrix will be tuned
                        )

                    else:
                        init_module_weights(
                            target.lora_A.default, sigma=0.00001)

            else:
                raise NotImplementedError(
                    "The only supported mode is: separated.")

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
