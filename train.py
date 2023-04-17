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

"""Training script."""

import functools
import os
import time

import gin
import numpy as np
import torch
from absl import app
from optree import tree_map
from torch.utils import tensorboard

from multinerf import configs, datasets, image, models, train_utils, utils, vis

configs.define_common_flags()
# jax.config.parse_flags_with_absl()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.


def main(unused_argv):
    # Shift the numpy random seed by host_id() to shuffle data loaded by different
    # hosts.
    np.random.seed(20201473)

    config = configs.load_config()

    dataset = datasets.load_dataset("train", config.data_dir, config)
    test_dataset = datasets.load_dataset("test", config.data_dir, config)

    cameras = dataset.cameras

    if config.rawnerf_mode:
        postprocess_fn = test_dataset.metadata["postprocess_fn"]
    else:
        postprocess_fn = lambda z, _=None: z

    model, optim, render_eval_fn, train_step, lr_fn = train_utils.setup_model(
        config, dataset=dataset
    )
    model = model.to("cuda")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters being optimized: {num_params}")

    if dataset.size > model.num_glo_embeddings and model.num_glo_features > 0:
        raise ValueError(
            f"Number of glo embeddings {model.num_glo_embeddings} "
            f"must be at least equal to number of train images "
            f"{dataset.size}"
        )

    metric_harness = image.MetricHarness()

    if not utils.isdir(config.checkpoint_dir):
        utils.makedirs(config.checkpoint_dir)
    checkpoint_path = os.path.join(config.checkpoint_dir, "ckpt.th")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        optim.load_state_dict(state_dict["optim"])
        model.load_state_dict(state_dict["model"])
        state_step = state_dict["step"]
    else:
        state_step = -1
    # Resume training at the step of the last checkpoint.
    init_step = state_step + 1

    # Set up TensorBoard logging on host 0.
    summary_writer = tensorboard.SummaryWriter(config.checkpoint_dir)
    if config.rawnerf_mode:
        for name, data in zip(["train", "test"], [dataset, test_dataset]):
            # Log shutter speed metadata in TensorBoard for debug purposes.
            for key in [
                "exposure_idx",
                "exposure_values",
                "unique_shutters",
            ]:
                summary_writer.text(f"{name}_{key}", str(data.metadata[key]), 0)

    total_time = 0
    total_steps = 0
    reset_stats = True
    if config.early_exit_steps is not None:
        num_steps = config.early_exit_steps
    else:
        num_steps = config.max_steps
    for step, batch in zip(range(init_step, num_steps + 1), dataset):
        model.train()
        batch = batch.apply_fn(lambda x: x.to("cuda"))

        if reset_stats:
            stats_buffer = []
            train_start_time = time.time()
            reset_stats = False

        learning_rate = lr_fn(step)
        train_frac = np.clip((step - 1) / (config.max_steps - 1), 0, 1)

        stats = train_step(batch, cameras, train_frac)

        for param in optim.param_groups:
            param["lr"] = learning_rate
        optim.zero_grad()
        stats["loss"].backward()
        optim.step()

        # Log training summaries. This is put behind a host_id check because in
        # multi-host evaluation, all hosts need to run inference even though we
        # only use host 0 to record results.
        stats_buffer.append(stats)

        if step == init_step or step % config.print_every == 0:
            elapsed_time = time.time() - train_start_time
            steps_per_sec = config.print_every / elapsed_time
            rays_per_sec = config.batch_size * steps_per_sec

            # A robust approximation of total training time, in case of pre-emption.
            total_time += int(round(TIME_PRECISION * elapsed_time))
            total_steps += config.print_every
            approx_total_time = int(round(step * total_time / total_steps))

            # Transpose and stack stats_buffer along axis 0.
            fs = [utils.flatten_dict(s, sep="/") for s in stats_buffer]
            stats_stacked = {
                k: torch.stack([f[k] for f in fs]) for k in fs[0].keys()
            }

            # Split every statistic that isn't a vector into a set of statistics.
            stats_split = {}
            for k, v in stats_stacked.items():
                if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
                    raise ValueError(
                        "statistics must be of size [n], or [n, k]."
                    )
                if v.ndim == 1:
                    stats_split[k] = v
                elif v.ndim == 2:
                    for i, vi in enumerate(tuple(v.T)):
                        stats_split[f"{k}/{i}"] = vi

            # Summarize the entire histogram of each statistic.
            for k, v in stats_split.items():
                summary_writer.add_histogram("train_" + k, v, step)

            # Take the mean and max of each statistic since the last summary.
            avg_stats = {k: torch.mean(v) for k, v in stats_split.items()}
            max_stats = {k: torch.max(v) for k, v in stats_split.items()}

            summ_fn = lambda s, v: summary_writer.add_scalar(
                s, v, step
            )  # pylint:disable=cell-var-from-loop

            # Summarize the mean and max of each statistic.
            for k, v in avg_stats.items():
                summ_fn(f"train_avg_{k}", v)
            for k, v in max_stats.items():
                summ_fn(f"train_max_{k}", v)

            summ_fn("train_num_params", num_params)
            summ_fn("train_learning_rate", learning_rate)
            summ_fn("train_steps_per_sec", steps_per_sec)
            summ_fn("train_rays_per_sec", rays_per_sec)

            summary_writer.add_scalar(
                "train_avg_psnr_timed",
                avg_stats["psnr"],
                total_time // TIME_PRECISION,
            )
            summary_writer.add_scalar(
                "train_avg_psnr_timed_approx",
                avg_stats["psnr"],
                approx_total_time // TIME_PRECISION,
            )

            if dataset.metadata is not None and model.learned_exposure_scaling:
                scalings = model.exposure_scaling_offsets.weight[0]
                num_shutter_speeds = dataset.metadata["unique_shutters"].shape[
                    0
                ]
                for i_s in range(num_shutter_speeds):
                    for j_s, value in enumerate(scalings[i_s]):
                        summary_name = f"exposure/scaling_{i_s}_{j_s}"
                        summary_writer.add_scalar(summary_name, value, step)

            precision = int(np.ceil(np.log10(config.max_steps))) + 1
            avg_loss = avg_stats["loss"]
            avg_psnr = avg_stats["psnr"]
            str_losses = (
                {  # Grab each "losses_{x}" field and print it as "x[:4]".
                    k[7:11]: (
                        f"{v:0.5f}" if v >= 1e-4 and v < 10 else f"{v:0.1e}"
                    )
                    for k, v in avg_stats.items()
                    if k.startswith("losses/")
                }
            )
            print(
                f"{step:{precision}d}"
                + f"/{config.max_steps:d}: "
                + f"loss={avg_loss:0.5f}, "
                + f"psnr={avg_psnr:6.3f}, "
                + f"lr={learning_rate:0.2e} | "
                + ", ".join([f"{k}={s}" for k, s in str_losses.items()])
                + f", {rays_per_sec:0.0f} r/s"
            )

            # Reset everything we are tracking between summarizations.
            reset_stats = True

        if step == 1 or step % config.checkpoint_every == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                },
                checkpoint_path,
            )

        # Test-set evaluation.
        if (
            config.train_render_every > 0
            and step % config.train_render_every == 0
        ):
            model.eval()

            # We reuse the same random number generator from the optimization step
            # here on purpose so that the visualization matches what happened in
            # training.
            eval_start_time = time.time()
            test_case = next(test_dataset)
            test_case = test_case.apply_fn(lambda x: x.to("cuda"))

            with torch.no_grad():
                rendering = models.render_image(
                    functools.partial(render_eval_fn, train_frac),
                    test_case.rays,
                    config,
                )

            # Log eval summaries on host 0.
            eval_time = time.time() - eval_start_time
            num_rays = np.prod(np.array(test_case.rays.directions.shape[:-1]))
            rays_per_sec = num_rays / eval_time
            summary_writer.add_scalar("test_rays_per_sec", rays_per_sec, step)
            print(
                f"Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec"
            )

            metric_start_time = time.time()
            with torch.no_grad():
                metric = metric_harness(
                    postprocess_fn(rendering["rgb"]),
                    postprocess_fn(test_case.rgb),
                )
            print(
                f"Metrics computed in {(time.time() - metric_start_time):0.3f}s"
            )
            for name, val in metric.items():
                if not np.isnan(val):
                    print(f"{name} = {val:.4f}")
                    summary_writer.add_scalar(
                        "train_metrics/" + name, val, step
                    )
            # move things to cpu for visualization
            if config.vis_decimate > 1:
                d = config.vis_decimate
                decimate_fn = (
                    lambda x, d=d: None if x is None else x[::d, ::d].cpu()
                )
            else:
                decimate_fn = lambda x: x.cpu()
            rendering = tree_map(decimate_fn, rendering)
            test_case = test_case.apply_fn(decimate_fn)
            vis_start_time = time.time()
            vis_suite = vis.visualize_suite(rendering, test_case.rays)
            print(f"Visualized in {(time.time() - vis_start_time):0.3f}s")
            if config.rawnerf_mode:
                # Unprocess raw output.
                vis_suite["color_raw"] = rendering["rgb"]
                # Autoexposed colors.
                vis_suite["color_auto"] = postprocess_fn(rendering["rgb"], None)
                summary_writer.add_image(
                    "test_true_auto",
                    postprocess_fn(test_case.rgb, None),
                    step,
                    dataformats="HWC",
                )
                # Exposure sweep colors.
                exposures = test_dataset.metadata["exposure_levels"]
                for p, x in list(exposures.items()):
                    vis_suite[f"color/{p}"] = postprocess_fn(
                        rendering["rgb"], x
                    )
                    summary_writer.add_image(
                        f"test_true_color/{p}",
                        postprocess_fn(test_case.rgb, x),
                        step,
                        dataformats="HWC",
                    )
            summary_writer.add_image(
                "test_true_color", test_case.rgb, step, dataformats="HWC"
            )
            if config.compute_normal_metrics:
                summary_writer.add_image(
                    "test_true_normals",
                    test_case.normals / 2.0 + 0.5,
                    step,
                    dataformats="HWC",
                )
            for k, v in vis_suite.items():
                summary_writer.add_image(
                    "test_output_" + k,
                    v,
                    step,
                    dataformats="HWC" if v.ndim == 3 else "HW",
                )

    # On host 0
    if config.max_steps % config.checkpoint_every != 0:
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            },
            checkpoint_path,
        )


if __name__ == "__main__":
    with gin.config_scope("train"):
        app.run(main)
