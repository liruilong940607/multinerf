# Copyright 2022 Google LLC
# Modified by Ruilong Li
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

"""Mathy utility functions."""

import numpy as np
import torch


def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def log_lerp(t, v0, v1):
    """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
    if v0 <= 0 or v1 <= 0:
        raise ValueError(f"Interpolants {v0} and {v1} must be positive.")
    lv0 = np.log(v0)
    lv1 = np.log(v1)
    return np.exp(clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(
    step, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1
):
    """Continuous learning rate decay function.
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    Args:
      step: int, the current optimization step.
      lr_init: float, the initial learning rate.
      lr_final: float, the final learning rate.
      max_steps: int, the number of steps during optimization.
      lr_delay_steps: int, the number of steps to delay the full learning rate.
      lr_delay_mult: float, the multiplier on the rate when delaying it.
    Returns:
      lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)


def interp(x, xp, fp):
    """A faster version of sorted_interp(), where xp and fp must be sorted."""
    indices = torch.searchsorted(xp.contiguous(), x.contiguous(), side="right")
    below = torch.clamp(indices - 1, 0, xp.shape[-1] - 1)
    above = torch.clamp(indices, 0, xp.shape[-1] - 1)
    fp0, fp1 = fp.gather(-1, below), fp.gather(-1, above)
    xp0, xp1 = xp.gather(-1, below), xp.gather(-1, above)

    offset = torch.clamp(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret


def sorted_interp(x, xp, fp):
    """A TPU-friendly version of interp(), where xp and fp must be sorted."""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(
            torch.where(mask, x[..., None], x[..., :1, None]), -2
        ).values
        x1 = torch.min(
            torch.where(~mask, x[..., None], x[..., -1:, None]), -2
        ).values
        return x0, x1

    fp0, fp1 = find_interval(fp)
    xp0, xp1 = find_interval(xp)

    offset = torch.clamp(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret
