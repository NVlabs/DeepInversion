#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import json
import re
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data

from fl_gradient_inversion import FLGradientInversion
from torchvision_class import TorchVisionClassificationModel


def run(cfg):
    """Run the gradient inversion attack.

    Args:
        cfg: Configuration dictionary containing the following keys used
            in to set up the attack. Should also contain the keys expected by
            FLGradientInversion's __call__() function.
            - model_name: Used to select the model aritechture,
            e.g. "resnet18".
            - num_classes:
            - pretrained:
            - checkpoint_file:
            - weights_file:
            - batchnorm_file:
    Returns:
        Reconstructed images.
    """
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TorchVisionClassificationModel(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
    )

    checkpoint_file = cfg["checkpoint_file"]
    add_weights = cfg["weights_file"]
    batchnorm_file = cfg["batchnorm_file"]
    input_parameters = []
    updates = np.load(add_weights, allow_pickle=True)["weights"].item()
    update_sum = 0.0
    n_excluded = 0
    weights = []
    if checkpoint_file:
        model_data = torch.load(checkpoint_file)
        if "model" in model_data.keys():
            net.load_state_dict(model_data["model"])
        else:
            net.load_state_dict(model_data)
    exclude_vars = None
    if exclude_vars:
        re_pattern = re.compile(exclude_vars)
    for name, _ in net.named_parameters():
        if exclude_vars:
            if re_pattern.search(name):
                n_excluded += 1
                weights.append(0.0)
            else:
                weights.append(1.0)
        val = updates[name]
        update_sum += np.sum(np.abs(val))
        val = torch.from_numpy(val).to(device)
        input_parameters.append(val)
    assert update_sum > 0.0, "All updates are zero!"
    model_bn = deepcopy(net).cuda()
    update_sum = 0.0
    new_state_dict = model_bn.state_dict()
    for n in updates.keys():
        val = updates[n]
        update_sum += np.sum(np.abs(val))
        new_state_dict[n] += torch.tensor(
            val, dtype=new_state_dict[n].dtype, device=new_state_dict[n].device
        )
    model_bn.load_state_dict(new_state_dict)
    assert update_sum > 0.0, "All updates are zero!"
    n_bn_updated = 0
    global_state_dict = net.state_dict()
    if batchnorm_file:
        bn_momentum = 0.1
        print(
            f"Using full BN stats from {batchnorm_file} "
            f"with momentum {bn_momentum} ! \n"
        )
        bn_stats = np.load(batchnorm_file, allow_pickle=True)["batchnorm"].item()
        for n in bn_stats.keys():
            if "running" in n:
                xt = (
                    bn_stats[n] - (1 - bn_momentum) * global_state_dict[n].numpy()
                ) / bn_momentum
                n_bn_updated += 1
                bn_stats[n] = xt

    net = net.to(device)
    grad_lst = []
    grad_lst_orig = np.load(add_weights, allow_pickle=True)["weights"].item()
    for name, _ in net.named_parameters():
        val = torch.from_numpy(grad_lst_orig[name]).cuda()
        grad_lst.append([name, val])
    grad_inversion_engine = FLGradientInversion(
        network=net,
        grad_lst=grad_lst,
        bn_stats=bn_stats,
        model_bn=model_bn,
    )
    grad_inversion_engine(cfg)


def main():
    with open("./config/config_inversion.json", "r") as f:
        cfg = json.load(f)

    run(cfg)


if __name__ == "__main__":
    main()
