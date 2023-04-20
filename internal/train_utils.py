# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training step and model creation functions."""

import collections
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Text, Tuple

import torch

from . import (
    camera_utils,
    configs,
    datasets,
    image,
    math,
    models,
    ref_utils,
    stepfun,
    utils,
)


def compute_data_loss(batch, renderings, rays, config):
    """Computes data loss terms for RGB, normal, and depth outputs."""
    data_losses = []
    stats = collections.defaultdict(lambda: [])

    # lossmult can be used to apply a weight to each ray in the batch.
    # For example: masking out rays, applying the Bayer mosaic mask, upweighting
    # rays from lower resolution images and so on.
    lossmult = rays.lossmult
    lossmult = torch.broadcast_to(lossmult, batch.rgb[..., :3].shape)
    if config.disable_multiscale_loss:
        lossmult = torch.ones_like(lossmult)

    for rendering in renderings:
        resid_sq = (rendering["rgb"] - batch.rgb[..., :3]) ** 2
        denom = lossmult.sum()
        stats["mses"].append((lossmult * resid_sq).sum() / denom)

        if config.data_loss_type == "mse":
            # Mean-squared error (L2) loss.
            data_loss = resid_sq
        elif config.data_loss_type == "charb":
            # Charbonnier loss.
            data_loss = torch.sqrt(resid_sq + config.charb_padding**2)
        elif config.data_loss_type == "rawnerf":
            # Clip raw values against 1 to match sensor overexposure behavior.
            rgb_render_clip = torch.clamp(rendering["rgb"], max=1.0)
            resid_sq_clip = (rgb_render_clip - batch.rgb[..., :3]) ** 2
            # Scale by gradient of log tonemapping curve.
            scaling_grad = 1.0 / (1e-3 + rgb_render_clip.detach())
            # Reweighted L2 loss.
            data_loss = resid_sq_clip * scaling_grad**2
        else:
            assert False
        data_losses.append((lossmult * data_loss).sum() / denom)

        if config.compute_disp_metrics:
            # Using mean to compute disparity, but other distance statistics can
            # be used instead.
            disp = 1 / (1 + rendering["distance_mean"])
            stats["disparity_mses"].append(((disp - batch.disps) ** 2).mean())

        if config.compute_normal_metrics:
            if "normals" in rendering:
                weights = rendering["acc"] * batch.alphas
                normalized_normals_gt = ref_utils.l2_normalize(batch.normals)
                normalized_normals = ref_utils.l2_normalize(
                    rendering["normals"]
                )
                normal_mae = ref_utils.compute_weighted_mae(
                    weights, normalized_normals, normalized_normals_gt
                )
            else:
                # If normals are not computed, set MAE to NaN.
                normal_mae = torch.nan
            stats["normal_maes"].append(normal_mae)

    data_losses = torch.stack(data_losses)
    loss = (
        config.data_coarse_loss_mult * torch.sum(data_losses[:-1])
        + config.data_loss_mult * data_losses[-1]
    )
    stats = {k: torch.stack(stats[k]) for k in stats}
    return loss, stats


def interlevel_loss(ray_history, config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    # Stop the gradient from the interlevel loss onto the NeRF MLP.
    last_ray_results = ray_history[-1]
    c = last_ray_results["sdist"].detach()
    w = last_ray_results["weights"].detach()
    loss_interlevel = 0.0
    for ray_results in ray_history[:-1]:
        cp = ray_results["sdist"]
        wp = ray_results["weights"]
        loss_interlevel += torch.mean(stepfun.lossfun_outer(c, w, cp, wp))
    return config.interlevel_loss_mult * loss_interlevel


def distortion_loss(ray_history, config):
    """Computes the distortion loss regularizer defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results["sdist"]
    w = last_ray_results["weights"]
    loss = torch.mean(stepfun.lossfun_distortion(c, w))
    return config.distortion_loss_mult * loss


def orientation_loss(rays, model, ray_history, config):
    """Computes the orientation loss regularizer defined in ref-NeRF."""
    total_loss = 0.0
    for i, ray_results in enumerate(ray_history):
        w = ray_results["weights"]
        n = ray_results[config.orientation_loss_target]
        if n is None:
            raise ValueError(
                "Normals cannot be None if orientation loss is on."
            )
        # Negate viewdirs to represent normalized vectors from point to camera.
        v = -1.0 * rays.viewdirs
        n_dot_v = (n * v[..., None, :]).sum(axis=-1)
        loss = torch.mean((w * torch.clamp(n_dot_v, max=0.0) ** 2).sum(axis=-1))
        if i < model.num_levels - 1:
            total_loss += config.orientation_coarse_loss_mult * loss
        else:
            total_loss += config.orientation_loss_mult * loss
    return total_loss


def predicted_normal_loss(model, ray_history, config):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    total_loss = 0.0
    for i, ray_results in enumerate(ray_history):
        w = ray_results["weights"]
        n = ray_results["normals"]
        n_pred = ray_results["normals_pred"]
        if n is None or n_pred is None:
            raise ValueError(
                "Predicted normals and gradient normals cannot be None if "
                "predicted normal loss is on."
            )
        loss = torch.mean(
            (w * (1.0 - torch.sum(n * n_pred, axis=-1))).sum(axis=-1)
        )
        if i < model.num_levels - 1:
            total_loss += config.predicted_normal_coarse_loss_mult * loss
        else:
            total_loss += config.predicted_normal_loss_mult * loss
    return total_loss


def create_train_step(
    model: models.Model,
    config: configs.Config,
    dataset: Optional[datasets.Dataset] = None,
):
    """Creates the pmap'ed Nerf training function.

    Args:
      model: The linen model.
      config: The configuration.
      dataset: Training dataset.

    Returns:
      pmap'ed training function.
    """
    if dataset is None:
        camtype = camera_utils.ProjectionType.PERSPECTIVE
    else:
        camtype = dataset.camtype

    def train_step(
        batch,
        cameras,
        train_frac,
    ):
        """One optimization step.

        Args:
          batch: dict, a mini-batch of data for training.
          cameras: module containing camera poses.
          train_frac: float, the fraction of training that is complete.

        Returns:
            stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        """

        def loss_fn():
            rays = batch.rays
            if config.cast_rays_in_train_step:
                rays = camera_utils.cast_ray_batch(cameras, rays, camtype)

            # Indicates whether we need to compute output normal or depth maps in 2D.
            compute_extras = (
                config.compute_disp_metrics or config.compute_normal_metrics
            )

            renderings, ray_history = model(
                config.randomized,
                rays,
                train_frac=train_frac,
                compute_extras=compute_extras,
                zero_glo=False,
            )

            losses = {}

            data_loss, stats = compute_data_loss(
                batch, renderings, rays, config
            )
            losses["data"] = data_loss

            if config.interlevel_loss_mult > 0:
                losses["interlevel"] = interlevel_loss(ray_history, config)

            if config.distortion_loss_mult > 0:
                losses["distortion"] = distortion_loss(ray_history, config)

            if (
                config.orientation_coarse_loss_mult > 0
                or config.orientation_loss_mult > 0
            ):
                losses["orientation"] = orientation_loss(
                    rays, model, ray_history, config
                )

            if (
                config.predicted_normal_coarse_loss_mult > 0
                or config.predicted_normal_loss_mult > 0
            ):
                losses["predicted_normals"] = predicted_normal_loss(
                    model, ray_history, config
                )

            stats["weight_l2s"] = {
                # naive weight decay:
                k: (p**2).sum()
                for k, p in model.named_parameters()
                # zipnerf uses this for ngp module:
                # k: (p**2).mean() for k, p in model.named_parameters()
            }

            if config.weight_decay_mults:
                losses["weight"] = sum(
                    [
                        config.weight_decay_mults.get(k.split(".")[0], 0.0) * v
                        for k, v in stats["weight_l2s"].items()
                    ]
                )

            stats["loss"] = sum(list(losses.values()))
            stats["losses"] = losses

            return stats

        stats = loss_fn()

        stats["psnrs"] = image.mse_to_psnr(stats["mses"])
        stats["psnr"] = stats["psnrs"][-1]
        return stats

    return train_step


def create_optimizer(
    config: configs.Config, model: models.Model
) -> Tuple[torch.optim.Optimizer, Callable[[int], float]]:
    """Creates optax optimizer for model training."""
    adam_kwargs = {
        "betas": [config.adam_beta1, config.adam_beta2],
        "eps": config.adam_eps,
    }
    lr_kwargs = {
        "max_steps": config.max_steps,
        "lr_delay_steps": config.lr_delay_steps,
        "lr_delay_mult": config.lr_delay_mult,
    }

    def get_lr_fn(lr_init, lr_final):
        return functools.partial(
            math.learning_rate_decay,
            lr_init=lr_init,
            lr_final=lr_final,
            **lr_kwargs,
        )

    lr_fn_main = get_lr_fn(config.lr_init, config.lr_final)
    optim = torch.optim.Adam(
        model.parameters(), lr=lr_fn_main(0), **adam_kwargs
    )

    return optim, lr_fn_main


def create_render_fn(model: models.Model):
    """Creates pmap'ed function for full image rendering."""

    def render_eval_fn(train_frac, rays):
        return model(
            False,  # Deterministic.
            rays,
            train_frac=train_frac,
            compute_extras=True,
        )

    return render_eval_fn


def setup_model(
    config: configs.Config,
    dataset: Optional[datasets.Dataset] = None,
) -> Tuple[
    models.Model,
    torch.optim.Optimizer,
    Callable[[float, utils.Rays], MutableMapping[Text, Any]],
    Callable[
        [utils.Batch, Optional[Tuple[Any, ...]], float],
        Dict[Text, Any],
    ],
    Callable[[int], float],
]:
    """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

    model = models.construct_model(config)
    optim, lr_fn = create_optimizer(config, model)
    render_eval_fn = create_render_fn(model)
    train_step = create_train_step(model, config, dataset=dataset)

    return model, optim, render_eval_fn, train_step, lr_fn
