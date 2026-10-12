# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Reconstruct one RGB-D object and export local splats and a camera-space mesh."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from pytorch3d.transforms import Transform3d, quaternion_to_matrix


# Change these variables to use another aligned RGB-D image and object mask.
tag = "hf"
input_dir = Path("notebook/images/nocs_0003_0354")
rgb_path = input_dir / "rgb.png"
depth_path = input_dir / "depth.png"
intrinsics_path = input_dir / "cam_K.txt"
mask_index = 0
seed = 42


def depth_to_pointmap(depth_mm, intrinsics):
    """Back-project millimeter depth into meters in PyTorch3D camera axes."""
    if depth_mm.ndim != 2:
        raise ValueError("Depth must be a single-channel H x W image.")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Camera intrinsics must be a finite 3 x 3 matrix.")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths fx and fy must be positive.")

    depth = depth_mm.astype(np.float32) / 1000.0
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    v, u = np.indices(depth.shape, dtype=np.float32)
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([-x, -y, depth], axis=-1).astype(np.float32)


def validate_inputs(image, depth, mask):
    """Reject misaligned inputs or a mask with no usable object depth."""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("RGB image must have three or four channels.")
    if depth.ndim != 2 or mask.ndim != 2:
        raise ValueError("Depth and mask must be single-channel H x W images.")
    if image.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise ValueError(
            "RGB, depth, and mask dimensions must match: "
            f"RGB={image.shape[:2]}, depth={depth.shape}, mask={mask.shape}."
        )
    if not mask.any():
        raise ValueError("Object mask is empty.")
    if not (mask & np.isfinite(depth) & (depth > 0)).any():
        raise ValueError("Object mask does not cover any valid depth pixels.")


def transform_mesh_vertices(vertices, rotation, translation, scale):
    """Apply the PR #79 mesh axes, scale, wxyz rotation, and translation."""
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device="cpu")
    rotation = rotation.detach().cpu().float().reshape(4)
    translation = translation.detach().cpu().float().reshape(3)
    scale = scale.detach().cpu().float().reshape(3)
    flip_z = vertices.new_tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    yup_to_zup = vertices.new_tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    pytorch3d_to_camera = vertices.new_tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    vertices = vertices @ flip_z @ yup_to_zup
    transform = (
        Transform3d(dtype=vertices.dtype, device=vertices.device)
        .scale(scale[None])
        .rotate(quaternion_to_matrix(rotation))
        .translate(translation[None])
    )
    return (transform.transform_points(vertices) @ pytorch3d_to_camera).numpy()


def main():
    # Use the same inference and image/mask loaders as demo.py.
    sys.path.append("notebook")
    from inference import Inference, load_image, load_single_mask

    image = load_image(str(rgb_path))
    mask = load_single_mask(str(input_dir), index=mask_index)
    # Do not use load_image here: it casts to uint8 and loses 16-bit depth.
    with Image.open(depth_path) as depth_image:
        depth = np.array(depth_image)
    intrinsics = np.loadtxt(intrinsics_path, dtype=np.float32)
    validate_inputs(image, depth, mask)
    pointmap = torch.from_numpy(depth_to_pointmap(depth, intrinsics))

    inference = Inference(f"checkpoints/{tag}/pipeline.yaml", compile=False)
    output = inference(image, mask, seed=seed, pointmap=pointmap)

    output["gs"].save_ply("splat_rgbd.ply")
    mesh = output["glb"].copy()
    mesh.vertices = transform_mesh_vertices(
        mesh.vertices, output["rotation"], output["translation"], output["scale"]
    )
    mesh.faces = mesh.faces[:, [0, 2, 1]]
    mesh.export("mesh_rgbd.glb")
    print("Your local Gaussian splat has been saved to splat_rgbd.ply")
    print("Your camera-space mesh (meters) has been saved to mesh_rgbd.glb")


if __name__ == "__main__":
    main()
