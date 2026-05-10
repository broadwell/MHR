import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from pygltflib import GLTF2

gltf = GLTF2()

glb_filename = sys.argv[1] # "../mhr_out/exported_animation.glb"
glb = GLTF2().load(glb_filename)

scene_idx = glb.scene if gltf.scene is not None else 0
scene = glb.scenes[scene_idx]

node_idx = scene.nodes[0]
node = glb.nodes[node_idx]

angle_y = 180
angle_z = 180

rot_y = R.from_euler('y', angle_y, degrees=True)
rot_z = R.from_euler('z', angle_z, degrees=True)

combined_rotation = rot_y * rot_z

quat = combined_rotation.as_quat()

node.rotation = quat.tolist()

outfn = glb_filename.replace(".glb", "_rotated.glb")

glb.save(outfn)
