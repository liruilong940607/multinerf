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

"""NeRF and its MLPs, with helper functions for construction and rendering."""

from typing import (
    Any,
    Callable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Text,
    Tuple,
)

import gin
import numpy as np
import torch
import torch.nn as nn
from optree import tree_map

from . import configs, coord, geopoly, image, ref_utils, render, stepfun, utils

gin.config.external_configurable(coord.contract, module="coord")


@gin.configurable
class Model(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""

    def __init__(
        self,
        config: Any = None,  # A Config class, must be set upon construction.
        num_prop_samples: int = 64,  # The number of samples for each proposal level.
        num_nerf_samples: int = 32,  # The number of samples the final nerf level.
        num_levels: int = (
            3  # The number of sampling levels (3==2 proposals, 1 nerf).
        ),
        bg_intensity_range: Tuple[float] = (
            1.0,
            1.0,
        ),  # The range of background colors.
        anneal_slope: float = 10,  # Higher = more rapid annealing.
        stop_level_grad: bool = True,  # If True, don't backprop across levels.
        use_viewdirs: bool = True,  # If True, use view directions as input.
        raydist_fn: Callable[..., Any] = None,  # The curve used for ray dists.
        ray_shape: str = "cone",  # The shape of cast rays ('cone' or 'cylinder').
        disable_integration: bool = False,  # If True, use PE instead of IPE.
        single_jitter: bool = True,  # If True, jitter whole rays instead of samples.
        dilation_multiplier: float = 0.5,  # How much to dilate intervals relatively.
        dilation_bias: float = 0.0025,  # How much to dilate intervals absolutely.
        num_glo_features: int = 0,  # GLO vector length, disabled if 0.
        num_glo_embeddings: int = 1000,  # Upper bound on max number of train images.
        learned_exposure_scaling: bool = (
            False  # Learned exposure scaling (RawNeRF).
        ),
        near_anneal_rate: Optional[
            float
        ] = None,  # How fast to anneal in near bound.
        near_anneal_init: float = (
            0.95  # Where to initialize near bound (in [0, 1]).
        ),
        single_mlp: bool = False,  # Use the NerfMLP for all rounds of sampling.
        resample_padding: float = 0.0,  # Dirichlet/alpha "padding" on the histogram.
        opaque_background: bool = False,  # If true, make the background opaque.
    ):
        super().__init__()
        self.config = config
        self.num_prop_samples = num_prop_samples
        self.num_nerf_samples = num_nerf_samples
        self.num_levels = num_levels
        self.bg_intensity_range = bg_intensity_range
        self.anneal_slope = anneal_slope
        self.stop_level_grad = stop_level_grad
        self.use_viewdirs = use_viewdirs
        self.raydist_fn = raydist_fn
        self.ray_shape = ray_shape
        self.disable_integration = disable_integration
        self.single_jitter = single_jitter
        self.dilation_multiplier = dilation_multiplier
        self.dilation_bias = dilation_bias
        self.num_glo_features = num_glo_features
        self.num_glo_embeddings = num_glo_embeddings
        self.learned_exposure_scaling = learned_exposure_scaling
        self.near_anneal_rate = near_anneal_rate
        self.near_anneal_init = near_anneal_init
        self.single_mlp = single_mlp
        self.resample_padding = resample_padding
        self.opaque_background = opaque_background

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP(
            num_glo_features=num_glo_features, use_viewdirs=use_viewdirs
        )
        self.prop_mlp = None if self.single_mlp else PropMLP(disable_rgb=True)

        if self.num_glo_features > 0:
            # Construct/grab GLO vectors for the cameras of each input ray.
            self.glo_embed = nn.Embedding(
                self.num_glo_embeddings, self.num_glo_features
            )
        else:
            self.glo_embed = None

        if self.learned_exposure_scaling:
            # Setup learned scaling factors for output colors.
            self.exposure_scaling_offsets = nn.Embedding(
                self.num_glo_embeddings, 3
            )
            # Initialize the learned scaling offsets at 0.
            self.exposure_scaling_offsets.weight.data.zero_()

    def forward(
        self,
        stratified,
        rays,
        train_frac,
        compute_extras,
        zero_glo=True,
    ):
        """The mip-NeRF Model.

        Args:
          stratified: False for deterministic output.
          rays: util.Rays, a pytree of ray origins, directions, and viewdirs.
          train_frac: float in [0, 1], what fraction of training is complete.
          compute_extras: bool, if True, compute extra quantities besides color.
          zero_glo: bool, if True, when using GLO pass in vector of zeros.

        Returns:
          ret: list, [*(rgb, distance, acc)]
        """
        if self.num_glo_features > 0:
            if not zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = rays.cam_idx[..., 0]
                glo_vec = self.glo_embed(cam_idx)
            else:
                glo_vec = torch.zeros(
                    rays.origins.shape[:-1] + (self.num_glo_features,),
                    device=rays.origins.device,
                )
        else:
            glo_vec = None

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(
            self.raydist_fn, rays.near, rays.far
        )

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        # `near_anneal_rate` can be used to anneal in the near bound at the start
        # of training, eg. 0.1 anneals in the bound over the first 10% of training.
        if self.near_anneal_rate is None:
            init_s_near = 0.0
        else:
            init_s_near = torch.clamp(
                1 - train_frac / self.near_anneal_rate, 0, self.near_anneal_init
            )
        init_s_far = 1.0
        sdist = torch.cat(
            [
                torch.full_like(rays.near, init_s_near),
                torch.full_like(rays.far, init_s_far),
            ],
            dim=-1,
        )
        weights = torch.ones_like(rays.near)
        prod_num_samples = 1

        ray_history = []
        renderings = []
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = (
                self.num_prop_samples if is_prop else self.num_nerf_samples
            )

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = (
                self.dilation_bias
                + self.dilation_multiplier
                * (init_s_far - init_s_near)
                / prod_num_samples
            )

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            use_dilation = (
                self.dilation_bias > 0 or self.dilation_multiplier > 0
            )
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True,
                )
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            # Optionally anneal the weights as a function of training iteration.
            if self.anneal_slope > 0:
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.0

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                -torch.inf,
            )

            # Draw sampled intervals from each ray's current weights.
            sdist = stepfun.sample_intervals(
                stratified,
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(init_s_near, init_s_far),
            )

            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.stop_level_grad:
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            gaussians = render.cast_rays(
                tdist,
                rays.origins,
                rays.directions,
                rays.radii,
                self.ray_shape,
                diag=False,
            )

            if self.disable_integration:
                # Setting the covariance of our Gaussian samples to 0 disables the
                # "integrated" part of integrated positional encoding.
                gaussians = (gaussians[0], torch.zeros_like(gaussians[1]))

            # Push our Gaussians through one of our two MLPs.
            mlp = (
                self.prop_mlp
                if (is_prop and not self.single_mlp)
                else self.nerf_mlp
            )
            ray_results = mlp(
                stratified,
                gaussians,
                viewdirs=rays.viewdirs if self.use_viewdirs else None,
                imageplane=rays.imageplane,
                glo_vec=None if is_prop else glo_vec,
                exposure=rays.exposure_values,
            )

            # Get the weights used by volumetric rendering (and our other losses).
            weights = render.compute_alpha_weights(
                ray_results["density"],
                tdist,
                rays.directions,
                opaque_background=self.opaque_background,
            )[0]

            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            elif not stratified:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (
                    self.bg_intensity_range[0] + self.bg_intensity_range[1]
                ) / 2
            else:
                # Sample RGB values from the range for each ray.
                bg_rgbs = (
                    torch.rand(
                        size=weights.shape[:-1] + (3,),
                        device=weights.device,
                    )
                    * (self.bg_intensity_range[1] - self.bg_intensity_range[0])
                    + self.bg_intensity_range[0]
                )

            # RawNeRF exposure logic.
            if rays.exposure_idx is not None:
                # Scale output colors by the exposure.
                ray_results["rgb"] *= rays.exposure_values[..., None, :]
                if self.learned_exposure_scaling:
                    exposure_idx = rays.exposure_idx[..., 0]
                    # Force scaling offset to always be zero when exposure_idx is 0.
                    # This constraint fixes a reference point for the scene's brightness.
                    mask = exposure_idx > 0
                    # Scaling is parameterized as an offset from 1.
                    scaling = 1 + mask[
                        ..., None
                    ] * self.exposure_scaling_offsets(exposure_idx)
                    ray_results["rgb"] *= scaling[..., None, :]

            # Render each ray.
            rendering = render.volumetric_rendering(
                ray_results["rgb"],
                weights,
                tdist,
                bg_rgbs,
                rays.far,
                compute_extras,
                extras={
                    k: v
                    for k, v in ray_results.items()
                    if k.startswith("normals") or k in ["roughness"]
                },
            )

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering["ray_sdist"] = sdist.reshape([-1, sdist.shape[-1]])[
                    :n, :
                ]
                rendering["ray_weights"] = weights.reshape(
                    [-1, weights.shape[-1]]
                )[:n, :]
                rgb = ray_results["rgb"]
                rendering["ray_rgbs"] = (rgb.reshape((-1,) + rgb.shape[-2:]))[
                    :n, :, :
                ]

            renderings.append(rendering)
            ray_results["sdist"] = sdist.clone()
            ray_results["weights"] = weights.clone()
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r["ray_weights"] for r in renderings]
            rgbs = [r["ray_rgbs"] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape)
                for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]["ray_rgbs"] = avg_rgbs[i]

        return renderings, ray_history


def construct_model(config):
    """Construct a mip-NeRF 360 model.

    Args:
      config: A Config class.

    Returns:
      model: initialized nn.Module, a NeRF model with parameters.
    """
    model = Model(config=config)
    return model


class MLP(nn.Module):
    """A PosEnc MLP."""

    def __init__(
        self,
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        bottleneck_width: int = 256,  # The width of the bottleneck vector.
        net_depth_viewdirs: int = 1,  # The depth of the second part of ML.
        net_width_viewdirs: int = 128,  # The width of the second part of MLP.
        net_activation: Callable[
            ..., Any
        ] = torch.relu,  # The activation function.
        min_deg_point: int = 0,  # Min degree of positional encoding for 3D points.
        max_deg_point: int = 12,  # Max degree of positional encoding for 3D points.
        skip_layer: int = (
            4  # Add a skip connection to the output of every N layers.
        ),
        skip_layer_dir: int = 4,  # Add a skip connection to 2nd MLP every N layers.
        num_rgb_channels: int = 3,  # The number of RGB channels.
        deg_view: int = 4,  # Degree of encoding for viewdirs or refdirs.
        use_viewdirs: bool = True,  # If True, use viewdirs or refdirs.
        use_reflections: bool = False,  # If True, use refdirs instead of viewdirs.
        use_directional_enc: bool = False,  # If True, use IDE to encode directions.
        # If False and if use_directional_enc is True, use zero roughness in IDE.
        enable_pred_roughness: bool = False,
        # Roughness activation function.
        roughness_activation: Callable[..., Any] = torch.nn.functional.softplus,
        roughness_bias: float = -1.0,  # Shift added to raw roughness pre-activation.
        use_diffuse_color: bool = (
            False  # If True, predict diffuse & specular colors.
        ),
        use_specular_tint: bool = False,  # If True, predict tint.
        use_n_dot_v: bool = False,  # If True, feed dot(n * viewdir) to 2nd MLP.
        bottleneck_noise: float = (
            0.0  # Std. deviation of noise added to bottleneck.
        ),
        density_activation: Callable[
            ..., Any
        ] = torch.nn.functional.softplus,  # Density activation.
        density_bias: float = -1.0,  # Shift added to raw densities pre-activation.
        density_noise: float = (
            0.0  # Standard deviation of noise added to raw density.
        ),
        rgb_premultiplier: float = 1.0,  # Premultiplier on RGB before activation.
        rgb_activation: Callable[
            ..., Any
        ] = torch.sigmoid,  # The RGB activation.
        rgb_bias: float = 0.0,  # The shift added to raw colors pre-activation.
        rgb_padding: float = 0.001,  # Padding added to the RGB outputs.
        enable_pred_normals: bool = False,  # If True compute predicted normals.
        disable_density_normals: bool = False,  # If True don't compute normals.
        disable_rgb: bool = False,  # If True don't output RGB.
        warp_fn: Callable[..., Any] = None,
        basis_shape: str = "icosahedron",  # `octahedron` or `icosahedron`.
        basis_subdivisions: int = (
            2  # Tesselation count. 'octahedron' + 1 == eye(3).
        ),
        num_glo_features: int = 0,  # Number of global features.
    ):
        super().__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.bottleneck_width = bottleneck_width
        self.net_depth_viewdirs = net_depth_viewdirs
        self.net_width_viewdirs = net_width_viewdirs
        self.net_activation = net_activation
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.skip_layer = skip_layer
        self.skip_layer_dir = skip_layer_dir
        self.num_rgb_channels = num_rgb_channels
        self.deg_view = deg_view
        self.use_viewdirs = use_viewdirs
        self.use_reflections = use_reflections
        self.use_directional_enc = use_directional_enc
        self.enable_pred_roughness = enable_pred_roughness
        self.roughness_activation = roughness_activation
        self.roughness_bias = roughness_bias
        self.use_diffuse_color = use_diffuse_color
        self.use_specular_tint = use_specular_tint
        self.use_n_dot_v = use_n_dot_v
        self.bottleneck_noise = bottleneck_noise
        self.density_activation = density_activation
        self.density_bias = density_bias
        self.density_noise = density_noise
        self.rgb_premultiplier = rgb_premultiplier
        self.rgb_activation = rgb_activation
        self.rgb_bias = rgb_bias
        self.rgb_padding = rgb_padding
        self.enable_pred_normals = enable_pred_normals
        self.disable_density_normals = disable_density_normals
        self.disable_rgb = disable_rgb
        self.warp_fn = warp_fn
        self.basis_shape = basis_shape
        self.basis_subdivisions = basis_subdivisions
        self.num_glo_features = num_glo_features

        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (
            self.enable_pred_normals or not self.disable_density_normals
        ):
            raise ValueError(
                "Normals must be computed for reflection directions."
            )

        # Precompute and store (the transpose of) the basis being used.
        pos_basis_t = torch.tensor(
            geopoly.generate_basis(
                self.basis_shape, self.basis_subdivisions
            ).copy(),
            dtype=torch.float32,
        ).T
        self.register_buffer("pos_basis_t", pos_basis_t, persistent=False)
        self.pos_enc_dim = (
            pos_basis_t.shape[-1]
            * 2
            * (self.max_deg_point - self.min_deg_point)
        )

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
        else:

            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction,
                    min_deg=0,
                    max_deg=self.deg_view,
                    append_identity=True,
                )

            self.dir_enc_fn = dir_enc_fn
        self.dir_enc_dim = self.dir_enc_fn(
            torch.empty((3,)), torch.empty((1,))
        ).shape[-1]

        # Setup the layers: density.
        density_layers = nn.ModuleList()
        in_features = self.pos_enc_dim
        for i in range(self.net_depth):
            density_layers.append(nn.Linear(in_features, self.net_width))
            if i % self.skip_layer == 0 and i > 0:
                in_features = self.net_width + self.pos_enc_dim
            else:
                in_features = self.net_width
        density_layers.append(nn.Linear(in_features, 1))
        self.density_layers = density_layers

        # Setup the layers: normals.
        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(self.pos_enc_dim, 3)

        # Setup the layers: colors.
        if self.disable_rgb:
            pass
        else:
            rgb_layers = nn.ModuleList()
            if self.use_viewdirs:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    self.diffuse_layer = nn.Linear(
                        self.pos_enc_dim, self.num_rgb_channels
                    )

                if self.use_specular_tint:
                    self.tint_layer = nn.Linear(self.pos_enc_dim, 3)

                if self.enable_pred_roughness:
                    self.roughness_layer = nn.Linear(self.pos_enc_dim, 1)

                # Output of the first part of MLP.
                input_dim = 0
                if self.bottleneck_width > 0:
                    self.bottleneck_layer = nn.Linear(
                        in_features, self.bottleneck_width
                    )
                    input_dim += self.bottleneck_width

                # Encode view (or reflection) directions.
                input_dim += self.dir_enc_dim

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    input_dim += 1

                # Append GLO vector if used.
                input_dim += self.num_glo_features

                # Output of the second part of MLP.
                in_features = input_dim
                for i in range(self.net_depth_viewdirs):
                    rgb_layers.append(
                        nn.Linear(in_features, self.net_width_viewdirs)
                    )
                    if i % self.skip_layer_dir == 0 and i > 0:
                        in_features = self.net_width_viewdirs + input_dim
                    else:
                        in_features = self.net_width_viewdirs

            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb_layers.append(nn.Linear(in_features, self.num_rgb_channels))
            self.rgb_layers = rgb_layers

        # initialize weights
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(
        self,
        stratified,
        gaussians,
        viewdirs=None,
        imageplane=None,
        glo_vec=None,
        exposure=None,
    ):
        """Evaluate the MLP.

        Args:
          stratified: False for no noise injection.
          gaussians: a tuple containing:                                           /
            - mean: [..., n, 3], coordinate means, and                             /
            - cov: [..., n, 3{, 3}], coordinate covariance matrices.
          viewdirs: torch.Tensor(float32), [..., 3], if not None, this variable will
            be part of the input to the second part of the MLP concatenated with the
            output vector of the first part of the MLP. If None, only the first part
            of the MLP will be used with input x. In the original paper, this
            variable is the view direction.
          imageplane: torch.Tensor(float32), [batch, 2], xy image plane coordinates
            for each ray in the batch. Useful for image plane operations such as a
            learned vignette mapping.
          glo_vec: [..., num_glo_features], The GLO vector for each ray.
          exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

        Returns:
          rgb: torch.Tensor(float32), with a shape of [..., num_rgb_channels].
          density: torch.Tensor(float32), with a shape of [...].
          normals: torch.Tensor(float32), with a shape of [..., 3], or None.
          normals_pred: torch.Tensor(float32), with a shape of [..., 3], or None.
          roughness: torch.Tensor(float32), with a shape of [..., 1], or None.
        """
        if not self.disable_rgb and self.use_viewdirs:
            assert viewdirs is not None
        if self.num_glo_features > 0:
            assert glo_vec is not None

        def predict_density(means, covs):
            """Helper function to output density."""
            # Encode input positions

            if self.warp_fn is not None:
                means, covs = coord.track_linearize(self.warp_fn, means, covs)

            lifted_means, lifted_vars = coord.lift_and_diagonalize(
                means, covs, self.pos_basis_t
            )
            x = coord.integrated_pos_enc(
                lifted_means,
                lifted_vars,
                self.min_deg_point,
                self.max_deg_point,
            )

            inputs = x
            # Evaluate network to produce the output density.
            for i in range(self.net_depth):
                x = self.density_layers[i](x)
                x = self.net_activation(x)
                if i % self.skip_layer == 0 and i > 0:
                    x = torch.cat([x, inputs], dim=-1)
            raw_density = self.density_layers[-1](x)[
                ..., 0
            ]  # Hardcoded to a single channel.

            # Add noise to regularize the density predictions if needed.
            if stratified and (self.density_noise > 0):
                raw_density = raw_density + self.density_noise * torch.randn(
                    raw_density.shape, device=raw_density.device
                )
            return raw_density, x

        means, covs = gaussians
        if self.disable_density_normals:
            raw_density, x = predict_density(means, covs)
            raw_grad_density = None
            normals = None
        else:
            # Flatten the input so value_and_grad can be vmap'ed.
            means_flat = means.reshape((-1, means.shape[-1]))
            covs_flat = covs.reshape((-1,) + covs.shape[len(means.shape) - 1 :])

            # Evaluate the network and its gradient on the flattened input.
            means_flat = means_flat.requires_grad_()
            raw_density_flat, x_flat = predict_density(means_flat, covs_flat)
            raw_grad_density_flat = torch.autograd.grad(
                raw_density_flat,
                means_flat,
                grad_outputs=torch.ones_like(raw_density_flat),
                retain_graph=True,
            )[0]

            # Unflatten the output.
            raw_density = raw_density_flat.reshape(means.shape[:-1])
            x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
            raw_grad_density = raw_grad_density_flat.reshape(means.shape)

            # Compute normal vectors as negative normalized density gradient.
            # We normalize the gradient of raw (pre-activation) density because
            # it's the same as post-activation density, but is more numerically stable
            # when the activation function has a steep or flat gradient.
            normals = -ref_utils.l2_normalize(raw_grad_density)

        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)

            # Normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals

        # Apply bias and activation to raw density
        density = self.density_activation(raw_density + self.density_bias)

        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros_like(means)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.tint_layer(x))

                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = self.roughness_activation(
                        raw_roughness + self.roughness_bias
                    )

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = self.bottleneck_layer(x)

                    # Add bottleneck noise.
                    if stratified and (self.bottleneck_noise > 0):
                        bottleneck = (
                            bottleneck
                            + self.bottleneck_noise
                            * torch.randn(
                                bottleneck.shape, device=bottleneck.device
                            )
                        )

                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(
                        -viewdirs[..., None, :], normals_to_use
                    )
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)

                    dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],),
                    )

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_to_use * viewdirs[..., None, :],
                        dim=-1,
                        keepdim=True,
                    )
                    x.append(dotprod)

                # Append GLO vector if used.
                if glo_vec is not None:
                    glo_vec = torch.broadcast_to(
                        glo_vec[..., None, :],
                        bottleneck.shape[:-1] + glo_vec.shape[-1:],
                    )
                    x.append(glo_vec)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)

                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.rgb_layers[i](x)
                    x = self.net_activation(x)
                    if i % self.skip_layer_dir == 0 and i > 0:
                        x = torch.cat([x, inputs], dim=-1)

            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = self.rgb_activation(
                self.rgb_premultiplier * self.rgb_layers[-1](x) + self.rgb_bias
            )

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clamp(
                    image.linear_to_srgb(specular_linear + diffuse_linear),
                    0.0,
                    1.0,
                )

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return dict(
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
        )


@gin.configurable
class NerfMLP(MLP):
    pass


@gin.configurable
class PropMLP(MLP):
    pass


def render_image(
    render_fn: Callable[
        [torch.Tensor, utils.Rays],
        Tuple[
            List[Mapping[Text, torch.Tensor]], List[Tuple[torch.Tensor, ...]]
        ],
    ],
    rays: utils.Rays,
    config: configs.Config,
    verbose: bool = True,
) -> MutableMapping[Text, Any]:
    """Render all the pixels of an image (in test mode).

    Args:
      render_fn: function, jit-ed render function mapping (rays) -> pytree.
      rays: a `Rays` pytree, the rays to be rendered.
      config: A Config class.
      verbose: print progress indicators.

    Returns:
      rgb: torch.Tensor, rendered color image.
      disp: torch.Tensor, rendered disparity image.
      acc: torch.Tensor, rendered accumulated weights per pixel.
    """
    height, width = rays.origins.shape[:2]
    num_rays = height * width
    rays = rays.apply_fn(lambda r: r.reshape((num_rays, -1)))

    chunks = []
    idx0s = range(0, num_rays, config.render_chunk_size)
    for i_chunk, idx0 in enumerate(idx0s):
        # pylint: disable=cell-var-from-loop
        if verbose and i_chunk % max(1, len(idx0s) // 10) == 0:
            print(f"Rendering chunk {i_chunk}/{len(idx0s)-1}")

        chunk_rays = rays.apply_fn(
            lambda r: r[idx0 : idx0 + config.render_chunk_size]
        )
        chunk_renderings, _ = render_fn(chunk_rays)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith("ray_"):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]
        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = tree_map(lambda *args: torch.cat(args), *chunks)
    for k, z in rendering.items():
        if not k.startswith("ray_"):
            # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])

    # After all of the ray bundles have been concatenated together, extract a
    # new random bundle (deterministically) from the concatenation that is the
    # same size as one of the individual bundles.
    keys = [k for k in rendering if k.startswith("ray_")]
    if keys:
        num_rays = rendering[keys[0]][0].shape[0]
        ray_idx = torch.randperm(num_rays, device=rays.origins.device)
        ray_idx = ray_idx[: config.vis_num_rays]
        for k in keys:
            rendering[k] = [r[ray_idx] for r in rendering[k]]

    return rendering
