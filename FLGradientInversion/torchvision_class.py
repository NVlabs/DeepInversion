# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        bias=True
    ):
        super().__init__()
        self.model = getattr(models, model_name)(pretrained=pretrained)
        if "fc" in dir(self.model):
            self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features,
                                            out_features=num_classes, bias=bias)
        elif "classifier" in dir(self.model) and "vgg" not in model_name:
            self.model.classifier = torch.nn.Linear(in_features=self.model.classifier.in_features,
                                                    out_features=num_classes, bias=bias)
        elif "vgg" in model_name:
            self.model.classifier[-1] = torch.nn.Linear(in_features=self.model.classifier[-1].in_features,
                                                        out_features=num_classes, bias=bias)
        else:
            raise ValueError(f"Model ['{model_name}'] does not have a supported classifier attribute.")

    def forward(self, x):
        return self.model(x)
