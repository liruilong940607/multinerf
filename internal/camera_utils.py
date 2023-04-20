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

"""Camera pose and ray generation utility functions."""

import enum
from typing import List, Mapping, Optional, Text, Tuple, Union

import numpy as np
import scipy
import torch
from torch import Tensor

from . import configs, stepfun, utils


def convert_to_ndc(
    origins: Tensor,
    directions: Tensor,
    pixtocam: Tensor,
    near: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Converts a set of rays to normalized device coordinates (NDC).

    Args:
      origins: Tensor(float32), [..., 3], world space ray origins.
      directions: Tensor(float32), [..., 3], world space ray directions.
      pixtocam: Tensor(float32), [3, 3], inverse intrinsic matrix.
      near: float, near plane along the negative z axis.

    Returns:
      origins_ndc: Tensor(float32), [..., 3].
      directions_ndc: Tensor(float32), [..., 3].

    This function assumes input rays should be mapped into the NDC space for a
    perspective projection pinhole camera, with identity extrinsic matrix (pose)
    and intrinsic parameters defined by inputs focal, width, and height.

    The near value specifies the near plane of the frustum, and the far plane is
    assumed to be infinity.

    The ray bundle for the identity pose camera will be remapped to parallel rays
    within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
    world space can be remapped as long as it has dz < 0 (ray direction has a
    negative z-coord); this allows us to share a common NDC space for "forward
    facing" scenes.

    Note that
        projection(origins + t * directions)
    will NOT be equal to
        origins_ndc + t * directions_ndc
    and that the directions_ndc are not unit length. Rather, directions_ndc is
    defined such that the valid near and far planes in NDC will be 0 and 1.

    See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
    """

    # Shift ray origins to near plane, such that oz = -near.
    # This makes the new near bound equal to 0.
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = torch.moveaxis(directions, -1, 0)
    ox, oy, oz = torch.moveaxis(origins, -1, 0)

    xmult = 1.0 / pixtocam[0, 2]  # Equal to -2. * focal / cx
    ymult = 1.0 / pixtocam[1, 2]  # Equal to -2. * focal / cy

    # Perspective projection into NDC for the t = 0 near points
    #     origins + 0 * directions
    origins_ndc = torch.stack(
        [xmult * ox / oz, ymult * oy / oz, -torch.ones_like(oz)], dim=-1
    )

    # Perspective projection into NDC for the t = infinity far points
    #     origins + infinity * directions
    infinity_ndc = torch.stack(
        [xmult * dx / dz, ymult * dy / dz, torch.ones_like(oz)], dim=-1
    )

    # directions_ndc points from origins_ndc to infinity_ndc
    directions_ndc = infinity_ndc - origins_ndc

    return origins_ndc, directions_ndc


def pad_poses(p: Tensor) -> Tensor:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = torch.broadcast_to(
        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=p.dtype, device=p.device),
        p[..., :1, :4].shape,
    )
    return torch.cat([p[..., :3, :4], bottom], dim=-2)


def unpad_poses(p: Tensor) -> Tensor:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def recenter_poses(poses: Tensor) -> Tuple[Tensor, Tensor]:
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = torch.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform


def average_pose(poses: Tensor) -> Tensor:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def viewmatrix(lookdir: Tensor, up: Tensor, position: Tensor) -> Tensor:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(torch.cross(up, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, position], dim=1)
    return m


def normalize(x: Tensor) -> Tensor:
    """Normalization helper function."""
    return x / torch.linalg.norm(x)


def focus_point_fn(poses: Tensor) -> Tensor:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = torch.eye(
        3, dtype=poses.dtype, device=poses.device
    ) - directions * torch.permute(directions, [0, 2, 1])
    mt_m = torch.permute(m, [0, 2, 1]) @ m
    focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


# Constants for generate_spiral_path():
NEAR_STRETCH = 0.9  # Push forward near bound for forward facing render path.
FAR_STRETCH = 5.0  # Push back far bound for forward facing render path.
FOCUS_DISTANCE = 0.75  # Relative weighting of near, far bounds for render path.


def generate_spiral_path(
    poses: Tensor,
    bounds: Tensor,
    n_frames: int = 120,
    n_rots: int = 2,
    zrate: float = 0.5,
) -> Tensor:
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of conservative near and far bounds in disparity space.
    near_bound = bounds.min().item() * NEAR_STRETCH
    far_bound = bounds.max().item() * FAR_STRETCH
    # All cameras will point towards the world space point (0, 0, -focal).
    focal = 1 / (
        ((1 - FOCUS_DISTANCE) / near_bound + FOCUS_DISTANCE / far_bound)
    )

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = torch.quantile(torch.abs(positions), 0.9, 0)
    radii = torch.cat([radii, torch.tensor([1.0], device=radii.device)])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(
        0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False
    ):
        t = radii * torch.tensor(
            [
                np.cos(theta),
                -np.sin(theta),
                -np.sin(theta * zrate),
                1.0,
            ],
            device=radii.device,
        )
        position = cam2world @ t
        lookat = cam2world @ torch.tensor(
            [0, 0, -focal, 1.0], device=radii.device
        )
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = torch.stack(render_poses, dim=0)
    return render_poses


def transform_poses_pca(poses: Tensor) -> Tuple[Tensor, Tensor]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
      poses: a (N, 3, 4) tensor containing the cameras' camera to world transforms.

    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    # torch.argsort, torch.linalg.det etc are not implemented for complex tensors
    # so we stick to numpy in this function.
    device = poses.device

    poses = poses.detach().cpu().numpy()

    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    # "sorting_kernel_method_name" not implemented for 'ComplexFloat'
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag([1.0, 1.0, -1.0]) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(
        transform @ pad_poses(torch.from_numpy(poses)).numpy()
    )
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag([1.0, -1.0, -1.0]) @ poses_recentered
        transform = np.diag([1.0, -1.0, -1.0, 1.0]) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag([scale_factor] * 3 + [1]) @ transform

    poses_recentered = torch.from_numpy(poses_recentered).float().to(device)
    transform = torch.from_numpy(transform).float().to(device)
    return poses_recentered, transform


def generate_ellipse_path(
    poses: Tensor,
    n_frames: int = 120,
    const_speed: bool = True,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> Tensor:
    """Generate an elliptical render path based on the given poses."""
    device = poses.device
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = torch.tensor([center[0], center[1], 0.0], device=device)

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = torch.quantile(torch.abs(poses[:, :3, 3] - offset), 0.9, dim=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = torch.quantile((poses[:, :3, 3]), 0.1, dim=0)
    z_high = torch.quantile((poses[:, :3, 3]), 0.9, dim=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return torch.stack(
            [
                low[0] + (high - low)[0] * (torch.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (torch.sin(theta) * 0.5 + 0.5),
                z_variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (torch.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                ),
            ],
            -1,
        )

    theta = torch.linspace(0, 2.0 * np.pi, n_frames + 1)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = torch.linalg.norm(positions[1:] - positions[:-1], dim=-1)
        theta = stepfun.sample(None, theta, torch.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / torch.linalg.norm(avg_up)
    ind_up = torch.argmax(torch.abs(avg_up))
    up = torch.eye(3)[ind_up] * torch.sign(avg_up[ind_up])

    return torch.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_interpolated_path(
    poses: Tensor,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """
    device = poses.device

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return torch.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return torch.tensor(
            [viewmatrix(p - l, u - p, p) for p, l, u in points], device=device
        )

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = torch.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = torch.tensor(scipy.interpolate.splev(u, tck))
        new_points = torch.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)


def interpolate_1d(
    x: Tensor, n_interp: int, spline_degree: int, smoothness: float
) -> Tensor:
    """Interpolate 1d signal x (by a factor of n_interp times)."""
    t = np.linspace(0, 1, len(x), endpoint=True)
    tck = scipy.interpolate.splrep(t, x, s=smoothness, k=spline_degree)
    n = n_interp * (len(x) - 1)
    u = np.linspace(0, 1, n, endpoint=False)
    return torch.tensor(scipy.interpolate.splev(u, tck), device=x.device)


def create_render_spline_path(
    config: configs.Config,
    image_names: Union[Text, List[Text]],
    poses: Tensor,
    exposures: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Creates spline interpolation render path from subset of dataset poses.

    Args:
      config: configs.Config object.
      image_names: either a directory of images or a text file of image names.
      poses: [N, 3, 4] array of extrinsic camera pose matrices.
      exposures: optional list of floating point exposure values.

    Returns:
      spline_indices: list of indices used to select spline keyframe poses.
      render_poses: array of interpolated extrinsic camera poses for the path.
      render_exposures: optional list of interpolated exposures for the path.
    """
    device = poses.device
    if utils.isdir(config.render_spline_keyframes):
        # If directory, use image filenames.
        keyframe_names = sorted(utils.listdir(config.render_spline_keyframes))
    else:
        # If text file, treat each line as an image filename.
        with utils.open_file(config.render_spline_keyframes, "r") as fp:
            # Decode bytes into string and split into lines.
            keyframe_names = fp.read().decode("utf-8").splitlines()
    # Grab poses corresponding to the image filenames.
    spline_indices = torch.tensor(
        [i for i, n in enumerate(image_names) if n in keyframe_names],
        device=device,
    )
    keyframes = poses[spline_indices]
    render_poses = generate_interpolated_path(
        keyframes,
        n_interp=config.render_spline_n_interp,
        spline_degree=config.render_spline_degree,
        smoothness=config.render_spline_smoothness,
        rot_weight=0.1,
    )
    if config.render_spline_interpolate_exposure:
        if exposures is None:
            raise ValueError(
                "config.render_spline_interpolate_exposure is True but "
                "create_render_spline_path() was passed exposures=None."
            )
        # Interpolate per-frame exposure value.
        log_exposure = torch.log(exposures[spline_indices])
        # Use aggressive smoothing for exposure interpolation to avoid flickering.
        log_exposure_interp = interpolate_1d(
            log_exposure,
            config.render_spline_n_interp,
            spline_degree=5,
            smoothness=20,
        )
        render_exposures = torch.exp(log_exposure_interp)
    else:
        render_exposures = None
    return spline_indices, render_poses, render_exposures


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> Tensor:
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return torch.tensor(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1.0],
        ],
        dtype=torch.float32,
    )


def get_pixtocam(focal: float, width: float, height: float) -> Tensor:
    """Inverse intrinsic matrix for a perfect pinhole camera."""
    camtopix = intrinsic_matrix(focal, focal, width * 0.5, height * 0.5)
    return torch.linalg.inv(camtopix)


def pixel_coordinates(width: int, height: int) -> Tuple[Tensor, Tensor]:
    """Tuple of the x and y integer coordinates for a grid of pixels."""
    return torch.meshgrid(
        torch.arange(width), torch.arange(height), indexing="xy"
    )


def _compute_residual_and_jacobian(
    x: Tensor,
    y: Tensor,
    xd: Tensor,
    yd: Tensor,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    assert k4 == 0.0, "k4 is not supported"
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * k3))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * 3.0 * k3)
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: Tensor,
    yd: Tensor,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    k4: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10,
) -> Tuple[Tensor, Tensor]:
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    x = xd.clone()
    y = yd.clone()

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(
            torch.abs(denominator) > eps,
            x_numerator / denominator,
            torch.zeros_like(denominator),
        )
        step_y = torch.where(
            torch.abs(denominator) > eps,
            y_numerator / denominator,
            torch.zeros_like(denominator),
        )

        x = x + step_x
        y = y + step_y

    return x, y


class ProjectionType(enum.Enum):
    """Camera projection type (standard perspective pinhole or fisheye model)."""

    PERSPECTIVE = "perspective"
    FISHEYE = "fisheye"


def pixels_to_rays(
    pix_x_int: Tensor,
    pix_y_int: Tensor,
    pixtocams: Tensor,
    camtoworlds: Tensor,
    distortion_params: Optional[Mapping[str, float]] = None,
    pixtocam_ndc: Optional[Tensor] = None,
    camtype: ProjectionType = ProjectionType.PERSPECTIVE,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Calculates rays given pixel coordinates, intrinisics, and extrinsics.

    Given 2D pixel coordinates pix_x_int, pix_y_int for cameras with
    inverse intrinsics pixtocams and extrinsics camtoworlds (and optional
    distortion coefficients distortion_params and NDC space projection matrix
    pixtocam_ndc), computes the corresponding 3D camera rays.

    Vectorized over the leading dimensions of the first four arguments.

    Args:
      pix_x_int: int tensor, shape SH, x coordinates of image pixels.
      pix_y_int: int tensor, shape SH, y coordinates of image pixels.
      pixtocams: float tensor, broadcastable to SH + [3, 3], inverse intrinsics.
      camtoworlds: float tensor, broadcastable to SH + [3, 4], camera extrinsics.
      distortion_params: dict of floats, optional camera distortion parameters.
      pixtocam_ndc: float tensor, [3, 3], optional inverse intrinsics for NDC.
      camtype: camera_utils.ProjectionType, fisheye or perspective camera.

    Returns:
      origins: float tensor, shape SH + [3], ray origin points.
      directions: float tensor, shape SH + [3], ray direction vectors.
      viewdirs: float tensor, shape SH + [3], normalized ray direction vectors.
      radii: float tensor, shape SH + [1], ray differential radii.
      imageplane: float tensor, shape SH + [2], xy coordinates on the image plane.
        If the image plane is at world space distance 1 from the pinhole, then
        imageplane will be the xy coordinates of a pixel in that space (so the
        camera ray direction at the origin would be (x, y, -1) in OpenGL coords).
    """

    # Must add half pixel offset to shoot rays through pixel centers.
    def pix_to_dir(x, y):
        return torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1)

    # We need the dx and dy rays to calculate ray radii for mip-NeRF cones.
    pixel_dirs_stacked = torch.stack(
        [
            pix_to_dir(pix_x_int, pix_y_int),
            pix_to_dir(pix_x_int + 1, pix_y_int),
            pix_to_dir(pix_x_int, pix_y_int + 1),
        ],
        dim=0,
    )

    # For jax, need to specify high-precision matmul.
    mat_vec_mul = lambda A, b: torch.matmul(A, b[..., None])[..., 0]

    # Apply inverse intrinsic matrices.
    camera_dirs_stacked = mat_vec_mul(pixtocams, pixel_dirs_stacked)

    if distortion_params is not None:
        # Correct for distortion.
        x, y = _radial_and_tangential_undistort(
            camera_dirs_stacked[..., 0],
            camera_dirs_stacked[..., 1],
            **distortion_params,
        )
        camera_dirs_stacked = torch.stack([x, y, torch.ones_like(x)], -1)

    if camtype == ProjectionType.FISHEYE:
        theta = torch.sqrt(
            torch.sum(torch.square(camera_dirs_stacked[..., :2]), dim=-1)
        )
        theta = torch.minimum(np.pi, theta)

        sin_theta_over_theta = torch.sin(theta) / theta
        camera_dirs_stacked = torch.stack(
            [
                camera_dirs_stacked[..., 0] * sin_theta_over_theta,
                camera_dirs_stacked[..., 1] * sin_theta_over_theta,
                torch.cos(theta),
            ],
            dim=-1,
        )

    # Flip from OpenCV to OpenGL coordinate system.
    camera_dirs_stacked = torch.matmul(
        camera_dirs_stacked, torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    )

    # Extract 2D image plane (x, y) coordinates.
    imageplane = camera_dirs_stacked[0, ..., :2]

    # Apply camera rotation matrices.
    directions_stacked = mat_vec_mul(
        camtoworlds[..., :3, :3], camera_dirs_stacked
    )
    # Extract the offset rays.
    directions, dx, dy = directions_stacked

    origins = torch.broadcast_to(camtoworlds[..., :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)

    if pixtocam_ndc is None:
        # Distance from each unit-norm direction vector to its neighbors.
        dx_norm = torch.linalg.norm(dx - directions, dim=-1)
        dy_norm = torch.linalg.norm(dy - directions, dim=-1)

    else:
        # Convert ray origins and directions into projective NDC space.
        origins_dx, _ = convert_to_ndc(origins, dx, pixtocam_ndc)
        origins_dy, _ = convert_to_ndc(origins, dy, pixtocam_ndc)
        origins, directions = convert_to_ndc(origins, directions, pixtocam_ndc)

        # In NDC space, we use the offset between origins instead of directions.
        dx_norm = torch.linalg.norm(origins_dx - origins, dim=-1)
        dy_norm = torch.linalg.norm(origins_dy - origins, dim=-1)

    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see the original mipnerf paper).
    radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / np.sqrt(12)

    return origins, directions, viewdirs, radii, imageplane


def cast_ray_batch(
    cameras: Tuple[Tensor, ...],
    pixels: utils.Pixels,
    camtype: ProjectionType = ProjectionType.PERSPECTIVE,
) -> utils.Rays:
    """Maps from input cameras and Pixel batch to output Ray batch.

    `cameras` is a Tuple of four sets of camera parameters.
      pixtocams: 1 or N stacked [3, 3] inverse intrinsic matrices.
      camtoworlds: 1 or N stacked [3, 4] extrinsic pose matrices.
      distortion_params: optional, dict[str, float] containing pinhole model
        distortion parameters.
      pixtocam_ndc: optional, [3, 3] inverse intrinsic matrix for mapping to NDC.

    Args:
      cameras: described above.
      pixels: integer pixel coordinates and camera indices, plus ray metadata.
        These fields can be an arbitrary batch shape.
      camtype: camera_utils.ProjectionType, fisheye or perspective camera.

    Returns:
      rays: Rays dataclass with computed 3D world space ray data.
    """
    pixtocams, camtoworlds, distortion_params, pixtocam_ndc = cameras

    # pixels.cam_idx has shape [..., 1], remove this hanging dimension.
    cam_idx = pixels.cam_idx[..., 0]
    batch_index = lambda arr: arr if arr.ndim == 2 else arr[cam_idx]

    # Compute rays from pixel coordinates.
    origins, directions, viewdirs, radii, imageplane = pixels_to_rays(
        pixels.pix_x_int,
        pixels.pix_y_int,
        batch_index(pixtocams),
        batch_index(camtoworlds),
        distortion_params=distortion_params,
        pixtocam_ndc=pixtocam_ndc,
        camtype=camtype,
    )

    # Create Rays data structure.
    return utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        imageplane=imageplane,
        lossmult=pixels.lossmult,
        near=pixels.near,
        far=pixels.far,
        cam_idx=pixels.cam_idx,
        exposure_idx=pixels.exposure_idx,
        exposure_values=pixels.exposure_values,
    )


def cast_pinhole_rays(
    camtoworld: Tensor,
    height: int,
    width: int,
    focal: float,
    near: float,
    far: float,
) -> utils.Rays:
    """Wrapper for generating a pinhole camera ray batch (w/o distortion)."""

    pix_x_int, pix_y_int = pixel_coordinates(width, height)
    pixtocam = get_pixtocam(focal, width, height)

    ray_args = pixels_to_rays(pix_x_int, pix_y_int, pixtocam, camtoworld)

    broadcast_scalar = lambda x: utils.broadcast_scalar(pix_x_int.shape, x)[
        ..., None
    ]
    ray_kwargs = {
        "lossmult": broadcast_scalar(1.0),
        "near": broadcast_scalar(near),
        "far": broadcast_scalar(far),
        "cam_idx": broadcast_scalar(0),
    }

    return utils.Rays(*ray_args, **ray_kwargs)


def cast_spherical_rays(
    camtoworld: Tensor,
    height: int,
    width: int,
    near: float,
    far: float,
) -> utils.Rays:
    """Generates a spherical camera ray batch."""

    theta_vals = torch.linspace(0, 2 * np.pi, width + 1)
    phi_vals = torch.linspace(0, np.pi, height + 1)
    theta, phi = torch.meshgrid(theta_vals, phi_vals, indexing="xy")

    # Spherical coordinates in camera reference frame (y is up).
    directions = torch.stack(
        [
            -torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
            torch.sin(phi) * torch.cos(theta),
        ],
        dim=-1,
    )

    # For jax, need to specify high-precision matmul.
    directions = torch.matmul(camtoworld[:3, :3], directions[..., None])[..., 0]

    dy = torch.diff(directions[:, :-1], dim=0)
    dx = torch.diff(directions[:-1, :], dim=1)
    directions = directions[:-1, :-1]
    viewdirs = directions

    origins = torch.broadcast_to(camtoworld[:3, -1], directions.shape)

    dx_norm = torch.linalg.norm(dx, dim=-1)
    dy_norm = torch.linalg.norm(dy, dim=-1)
    radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / np.sqrt(12)

    imageplane = torch.zeros_like(directions[..., :2])

    ray_args = (origins, directions, viewdirs, radii, imageplane)

    broadcast_scalar = lambda x: utils.broadcast_scalar(radii.shape, x)[
        ..., None
    ]
    ray_kwargs = {
        "lossmult": broadcast_scalar(1.0),
        "near": broadcast_scalar(near),
        "far": broadcast_scalar(far),
        "cam_idx": broadcast_scalar(0),
    }

    return utils.Rays(*ray_args, **ray_kwargs)
