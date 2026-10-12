# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os

import numpy as np
import trimesh
from pytorch3d.transforms import quaternion_to_matrix

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_masks, make_scene


def make_mesh_scene(*outputs):
    """Copy object meshes into the same scene coordinates as make_scene."""
    scene = trimesh.Scene()
    # Undo the row-vector Z-up to Y-up rotation in postprocessing_utils.to_glb.
    yup_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    for index, output in enumerate(outputs):
        mesh = output.get("glb")
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"Object {index} has no mesh for GLB export.")
        mesh = mesh.copy()
        scale = output["scale"].detach().cpu().numpy().reshape(3)
        rotation = (
            quaternion_to_matrix(output["rotation"].detach().cpu())
            .numpy()
            .reshape(3, 3)
        )
        translation = output["translation"].detach().cpu().numpy().reshape(3)

        # make_scene uses row vectors: scale, then rotate, then translate.
        # Trimesh uses column vectors, so transpose the combined linear map.
        transform = np.eye(4)
        transform[:3, :3] = (yup_to_zup @ np.diag(scale) @ rotation).T
        transform[:3, 3] = translation
        mesh.apply_transform(transform)
        name = f"object_{index}"
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    return scene


# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

IMAGE_PATH = "/data/tools/sam-3d-objects/data/cropped/color.png"
image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")

# run model
outputs = [inference(image, mask, seed=42) for mask in masks]

scene_gs = make_scene(*outputs)

scene_gs.save_ply(f"splat_multi_object.ply")
print("Your multi-object reconstruction has been saved to splat_multi_object.ply")

scene_mesh = make_mesh_scene(*outputs)
scene_mesh.export("mesh_multi_object.glb")
print("Your multi-object mesh has been saved to mesh_multi_object.glb")
