# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from monai.utils import optional_import

models, _ = optional_import("torchvision.models")


class TorchVisionClassificationModel(torch.nn.Module):
    """
    Customize TorchVision models to replace final linear/fully-connected layer to fit number of classes.

    Args:
        model_name: fully connected layer at the end from https://pytorch.org/vision/stable/models.html, e.g.
            ``resnet18`` (default), ``alexnet``, ``vgg16``, etc.
        num_classes: number of classes for the last classification layer. Default to 1.
        pretrained: whether to use the imagenet pretrained weights. Default to False.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 1,
        pretrained: bool = False,
        bias=True,
    ):
        super().__init__()
        self.model = getattr(models, model_name)(pretrained=pretrained)
        if "fc" in dir(self.model):
            self.model.fc = torch.nn.Linear(
                in_features=self.model.fc.in_features,
                out_features=num_classes,
                bias=bias,
            )
        elif "classifier" in dir(self.model) and "vgg" not in model_name:
            self.model.classifier = torch.nn.Linear(
                in_features=self.model.classifier.in_features,
                out_features=num_classes,
                bias=bias,
            )
        elif "vgg" in model_name:
            self.model.classifier[-1] = torch.nn.Linear(
                in_features=self.model.classifier[-1].in_features,
                out_features=num_classes,
                bias=bias,
            )
        else:
            raise ValueError(
                f"Model ['{model_name}'] does not have a supported classifier attribute."
            )

    def forward(self, x):
        return self.model(x)
