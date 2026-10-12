# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_masks, make_scene

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

IMAGE_PATH = "/data/tools/sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")

# run model
outputs = [inference(image, mask, seed=42) for mask in masks]

scene_gs = make_scene(*outputs)

scene_gs.save_ply(f"splat_multi_object.ply")
print("Your multi-object reconstruction has been saved to splat_multi_object.ply")