# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import collections
import json
import logging
import os
from copy import deepcopy
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ignite.engine import Engine
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.engines.utils import IterationEvents, default_prepare_batch
from monai.inferers import SimpleInferer
from monai.utils.enums import CommonKeys as Keys
from PIL import Image


class FLGradientInversion(object):
    def __init__(
        self,
        network,
        grad_lst,
        bn_stats,
        model_bn,
        prior_transforms=None,
        save_transforms=None,
    ):
        """FLGradientInversion is used to reconstruct training images and
        targets (ground truth labels) by attempting to invert the gradients
        (model updates) shared in a federated learning framework.

        Args:
            network: network for which the gradients are being inverted,
                i.e. the current global model the models updates are being
                computed with respect to.
            grad_lst: model updates.
            bn_stats: updated batch norm statistics.
            model_bn: updated model containing current batch norm statistics.
            prior_transforms: Optional custom transforms to read the prior
                image. Defaults to None.
            save_transforms: Optional transforms to save the reconstructed
                images. Defaults to None.
        Returns:
            __call__() function returns the reconstructions.
        """
        self.network = network
        self.bn_stats = bn_stats
        self.model_bn = model_bn
        self.loss_r_feature_layers = []
        self.grad_lst = grad_lst
        self.logger = logging.getLogger(self.__class__.__name__)
        self.prior_transforms = prior_transforms
        self.save_transforms = save_transforms

    def __call__(self, cfg):
        """Run the gradient inversion attack.

        Args:
            cfg: Configuration dictionary containing the following keys used
                in this call.
                - img_prior: full path to prior image file used to initialize
                    the attack.
                - save_path: Optional save directory where reconstructed
                    images and targets are being saved.
                - criterion: Loss used for training the classification
                    network, e.g. "BCEWithLogitsLoss".
                - iterations: number of iterations to run the attack.
                - resolution: x/y dimension of the images to be reconstructed.
                - start_rand: Whether to start from random initialization.
                    If `False`, the `img_prior` is used.
                - init_target_rand: Whether to initialize the reconstructed
                    targets using a uniform distribution. If `False`, targets
                    are initialized as all zeros.
                - no_lr_decay: Disable the learning rate decay of the
                    optimizer.
                - grad_l2: L2 scaling factor on the gradient loss.
                - original_bn_l2: Scaling factor for batchnorm matching loss.
                - energy_l2: This adds gaussian noise to find global minimums.
                - tv_l1: Coefficient for total variation L1 loss.
                - tv_l2: Coefficient for total variation L2 loss.
                - lr: Learning rate for optimization.
                - l2: L2 loss on the image.
                - local_epoch: Local number of epochs used by the FL client.
                - local_optim: Local optimizer used by the FL client, Either
                    "sgd" or "adam".
                - save_every: How often to save the reconstructions to file.
        Returns:
            Reconstructed images.
        """
        self.save_path = cfg["save_path"]
        save_every = cfg["save_every"]
        if save_every > 0:
            self.create_folder(self.save_path)

        if cfg["criterion"] == "BCEWithLogitsLoss":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif cfg["criterion"] == "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "criterion should be BCEWithLogitsLoss or CrossEntropyLoss."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = self.network
        local_rank = torch.cuda.current_device()
        if cfg["start_rand"]:
            inputs_1 = torch.randn(
                (cfg["batch_size"], 1, cfg["resolution"], cfg["resolution"]),
                requires_grad=True,
                device=device,
                dtype=torch.float,
            )
        else:
            prior_file = cfg["img_prior"]
            if self.prior_transforms:
                _img = self.prior_transforms(prior_file)
            else:  # use default prior loading transforms
                pil_img = Image.open(prior_file)
                self.prior_transforms = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(
                            (cfg["resolution"], cfg["resolution"])
                        ),
                        torchvision.transforms.ToTensor(),
                    ]
                )
                _img = self.prior_transforms(pil_img)

            # make init batch
            images = torch.empty(
                size=(
                    cfg["local_num_images"],
                    1,
                    cfg["resolution"],
                    cfg["resolution"],
                )
            )
            for i in range(cfg["local_num_images"]):
                images[i] = _img.unsqueeze_(0)
            inputs_1 = images.to(device)
            inputs_1.requires_grad_(True)

        if cfg["init_target_rand"]:
            targets_in = torch.rand(
                (cfg["local_num_images"], 2),
                requires_grad=True,
                device=device,
                dtype=torch.float,
            )
        else:
            targets_in = torch.zeros(
                (cfg["local_num_images"], 2),
                requires_grad=True,
                device=device,
                dtype=torch.float,
            )

        iteration = -1
        for lr_it, _ in enumerate([2, 1]):
            iterations_per_layer = cfg["iterations"]
            if lr_it == 0:
                continue
            optimizer = torch.optim.Adam(
                [inputs_1, targets_in],
                lr=cfg["lr"],
                betas=[0.9, 0.9],
                eps=1e-8,
            )
            lr_scheduler = self.lr_cosine_policy(cfg["lr"], 100, iterations_per_layer)
            local_trainer = self.create_trainer(
                cfg=cfg,
                network=network,
                inputs=(
                    inputs_1 * torch.ones((1, 3, 1, 1)).cuda()
                ),  # turn grayscale to RGB (3-channel inputs)
                targets=targets_in,
                criterion=criterion,
                device=torch.device("cuda"),
            )
            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                if not cfg["no_lr_decay"]:
                    lr_scheduler(optimizer, iteration_loc, iteration_loc)
                inputs = inputs_1 * torch.ones((1, 3, 1, 1)).cuda()
                optimizer.zero_grad()
                network.zero_grad()
                network.train()
                loss_var_l1, loss_var_l2 = self.img_prior(inputs)
                loss_l2 = torch.norm(
                    inputs.view(cfg["local_num_images"], -1), dim=1
                ).mean()
                loss_aux = (
                    cfg["tv_l2"] * loss_var_l2
                    + cfg["tv_l1"] * loss_var_l1
                    + cfg["l2"] * loss_l2
                )
                loss = loss_aux
                if cfg["grad_l2"] > 0:
                    new_grad = self.sim_local_updates(
                        cfg=cfg,
                        trainer=local_trainer,
                        network=network,
                        inputs=inputs,
                        targets=targets_in,
                        use_sigmoid=True,
                        use_softmax=False,
                    )
                    loss_grad = 0
                    for a, b in zip(new_grad, self.grad_lst):
                        loss_grad += cfg["grad_l2"] * (torch.norm(a - b[1]))
                    loss = loss + loss_grad

                # add batch norm loss
                bn_hooks = []
                self.model_bn.train()
                for name, module in self.model_bn.named_modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        bn_hooks.append(
                            DeepInversionFeatureHook(
                                module=module,
                                bn_stats=self.bn_stats,
                                name=name,
                            )
                        )
                # run forward path once to compute bn_hooks
                self.model_bn(inputs)
                loss_bn_tmp = 0
                for hook in bn_hooks:
                    loss_bn_tmp += hook.r_feature
                    hook.close()
                loss_bn = cfg["original_bn_l2"] * loss_bn_tmp
                loss += loss_bn
                loss.backward(retain_graph=True)
                optimizer.step()
                if local_rank == 0:
                    if iteration % save_every == 0:
                        self.logger.info(f"------------iteration {iteration}----------")
                        self.logger.info(f"total loss {loss.item()}")
                        self.logger.info(
                            f"mean targets {torch.mean(targets_in, 0).detach().cpu().numpy()}"
                        )
                        self.logger.info(f"gradient loss {loss_grad.item()}")
                        self.logger.info(f"bn matching loss {loss_bn.item()}")
                        self.logger.info(
                            f"tvl2 loss {cfg['tv_l2'] * loss_var_l2.item()}"
                        )
                best_inputs = inputs.clone()
                if iteration % save_every == 0 and (save_every > 0):
                    self.save_results(
                        images=best_inputs, targets=targets_in, name="recon"
                    )
                    # save reconstruction collage
                    torchvision.utils.save_image(
                        best_inputs,
                        os.path.join(self.save_path, "recon.png"),
                        normalize=True,
                        scale_each=True,
                        nrow=int(int(cfg["local_num_images"]) ** 0.5),
                    )
                if cfg["energy_l2"] > 0.0:
                    inputs_noise_add = torch.randn(inputs.size(), device=device)
                    for param_group in optimizer.param_groups:
                        current_lr = param_group["lr"]
                        break
                    std = cfg["energy_l2"] * current_lr
                    if iteration % save_every == 0:
                        if local_rank == 0:
                            self.logger.info(
                                f"Energy method waken up, "
                                f"adding Gaussian of std {std}"
                            )
                    inputs.data = inputs.data + inputs_noise_add * std

        if save_every > 0:
            self.save_results(images=best_inputs, targets=targets_in, name="recon")

        optimizer.state = collections.defaultdict(dict)

        return best_inputs, targets_in

    @staticmethod
    def sim_local_updates(
        cfg,
        trainer,
        network,
        inputs,
        targets,
        use_softmax=False,
        use_sigmoid=True,
    ):
        """
        Run the equivalent local optimization loop to get gradients
        which will be matched (using SupervisedTrainer)
        """
        trainer.logger.setLevel(logging.WARNING)

        params_before = deepcopy(network.state_dict())
        trainer.network.load_state_dict(params_before)
        if use_softmax and use_sigmoid:
            raise ValueError(
                "Only set one of `use_softmax` or `use_sigmoid` to be true."
            )
        if use_softmax:
            targets = torch.softmax(targets, dim=-1)
        if use_sigmoid:
            targets = torch.sigmoid(targets)
        data = []
        for i in range(cfg["local_num_images"]):
            data.append({Keys.IMAGE: inputs[i, ...], Keys.LABEL: targets[i, ...]})
        trainer.data_loader = DataLoader([data], batch_size=cfg["local_bs"])
        if cfg["local_optim"] == "sgd":
            optimizer = torch.optim.SGD(network.parameters(), cfg["lr_local"])
        elif cfg["local_optim"] == "adam":
            optimizer = torch.optim.Adam(network.parameters(), cfg["lr_local"])
        else:
            raise ValueError(
                f"Local optimizer {cfg['local_optim']} " f"is not currently supported !"
            )
        trainer.optimizer.load_state_dict(optimizer.state_dict())
        trainer.optimizer.zero_grad()
        trainer.network.zero_grad()
        trainer.run()
        params_after = trainer.network.state_dict()
        new_grad = []
        for name, _ in network.named_parameters():
            new_grad.append(params_after[name] - params_before[name])
        return new_grad

    @staticmethod
    def create_trainer(cfg, network, inputs, targets, criterion, device=None):
        if device is None:
            device = torch.device("cuda")

        data = []
        for i in range(cfg["local_num_images"]):
            data.append({Keys.IMAGE: inputs[i, ...], Keys.LABEL: targets[i, ...]})
        loader = DataLoader([data], batch_size=cfg["local_bs"])
        if cfg["local_optim"] == "sgd":
            optimizer = torch.optim.SGD(network.parameters(), cfg["lr_local"])
        elif cfg["local_optim"] == "adam":
            optimizer = torch.optim.Adam(network.parameters(), cfg["lr_local"])
        else:
            raise ValueError(
                "Local optimizer {} is not currently supported !".format(
                    cfg["local_optim"]
                )
            )
        optimizer.zero_grad()
        trainer = InversionSupervisedTrainer(
            device=device,
            max_epochs=cfg["local_epoch"],
            train_data_loader=loader,
            network=network,
            optimizer=optimizer,
            loss_function=criterion,
            amp=False,
        )
        return trainer

    def img_prior(self, inputs_jit):
        # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
        loss_var_l2 = (
            torch.norm(diff1)
            + torch.norm(diff2)
            + torch.norm(diff3)
            + torch.norm(diff4)
        )
        loss_var_l1 = (
            (diff1.abs() / 255.0).mean()
            + (diff2.abs() / 255.0).mean()
            + (diff3.abs() / 255.0).mean()
            + (diff4.abs() / 255.0).mean()
        )
        loss_var_l1 = loss_var_l1 * 255.0
        return loss_var_l1, loss_var_l2

    def denormalize(self, image_tensor, use_fp16=False):

        if use_fp16:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

        for c in range(3):
            m, s = mean[c], std[c]

            if len(image_tensor.shape) == 4:
                image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

            elif len(image_tensor.shape) == 3:
                image_tensor[c] = torch.clamp(image_tensor[c] * s + m, 0, 1)
            else:
                raise NotImplementedError()

        return image_tensor

    def create_folder(self, directory):

        if not os.path.exists(directory):
            os.makedirs(directory)

    def lr_policy(self, lr_fn):
        def _alr(optimizer, iteration, epoch):
            lr = lr_fn(iteration, epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        return _alr

    def lr_cosine_policy(self, base_lr, warmup_length, epochs):
        def _lr_fn(iteration, epoch):
            if epoch < warmup_length:
                lr = base_lr * (epoch + 1) / warmup_length
            else:
                e = epoch - warmup_length
                es = epochs - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            return lr

        return self.lr_policy(_lr_fn)

    def save_results(self, images, targets, name="recon"):
        # save reconstructed images
        for id in range(images.shape[0]):
            img = images[id, ...]
            if self.save_transforms:
                self.save_transforms(img)
            else:
                save_name = f"{name}_{id}.png"
                place_to_store = os.path.join(self.save_path, save_name)

                image_np = img.data.cpu().numpy()
                image_np = image_np.transpose((1, 2, 0))
                image_np = np.array(
                    (image_np - np.min(image_np))
                    / (np.max(image_np) - np.min(image_np))
                )
                plt.imsave(place_to_store, image_np)

        # save reconstructed targets
        place_to_store = os.path.join(self.save_path, f"{name}_targets.json")

        with open(place_to_store, "w") as f:
            json.dump(targets.detach().cpu().numpy().tolist(), f, indent=4)


class InversionSupervisedTrainer(SupervisedTrainer):
    """
    Same as MONAI's SupervisedTrainer but using
    retain_graph=True in backward() calls.
    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        amp: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            train_data_loader=train_data_loader,
            network=network,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=None,
            inferer=SimpleInferer(),
            key_train_metric=None,
            additional_metrics=None,
            amp=amp,
            event_names=None,
            event_to_attr=None,
        )

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        """
        Callback function for the Supervised Training processing logic of 1
        iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved
            to device.
            - LABEL: label Tensor data corresponding to the image, already
            moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary
            or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        def _compute_pred_loss():
            engine.state.output[Keys.PRED] = self.inferer(
                inputs, self.network, *args, **kwargs
            )
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            engine.state.output[Keys.LOSS] = self.loss_function(
                engine.state.output[Keys.PRED], targets
            ).mean()
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        self.network.train()
        self.network.zero_grad()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                _compute_pred_loss()
            self.scaler.scale(engine.state.output[Keys.LOSS]).backward(
                retain_graph=True
            )
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            _compute_pred_loss()
            engine.state.output[Keys.LOSS].backward(retain_graph=True)
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.optimizer.step()
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        return engine.state.output


class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and
    compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module, bn_stats=None, name=None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.bn_stats = bn_stats
        self.name = name
        self.r_feature = None
        self.mean = None
        self.var = None

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1, unbiased=False)
        )
        if self.bn_stats is None:
            var_feature = torch.norm(module.running_var.data - var, 2)
            mean_feature = torch.norm(module.running_mean.data - mean, 2)
        else:
            var_feature = torch.norm(
                torch.tensor(
                    self.bn_stats[self.name + ".running_var"], device=input[0].device
                )
                - var,
                2,
            )
            mean_feature = torch.norm(
                torch.tensor(
                    self.bn_stats[self.name + ".running_mean"], device=input[0].device
                )
                - mean,
                2,
            )

        rescale = 1.0
        self.r_feature = mean_feature + rescale * var_feature
        self.mean = mean
        self.var = var

    def close(self):
        self.hook.remove()
