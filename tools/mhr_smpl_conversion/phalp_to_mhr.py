# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
pixi run python phalp_to_mhr.py path/to/phalp/output.pkl
"""


import argparse
import os
import sys

import cv2
import joblib
import numpy as np
import smplx

import torch
import trimesh
from mhr.mhr import MHR
from conversion import Conversion

_OUTPUT_DIR = "./tmp_results"  # Directory to store conversion results

# Can change to match gender of dancer, if desired
#SMPL_MODEL_FILE = "/srv/mime/software/MHR/data/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
SMPL_MODEL_FILE = "/srv/mime/software/MHR/data/basicmodel_f_lbs_10_207_0_v1.1.0.pkl"
#SMPL_MODEL_FILE = "/srv/mime/software/MHR/data/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"

smpl_model = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_phalp_poses_file = sys.argv[1]
output_dir = _OUTPUT_DIR
smpl_model_file = SMPL_MODEL_FILE
        
if smpl_model_file is not None:
    # Unfortunately, SMPL model .pkl file may come with Chumpy, which is not compatible with
    # latest Python versions. Although the latest official SMPL model in .npz format is chumpy
    # free, the default smplx package does not support .npz file as SMPL model file.
    # So please provide either a chumpy-free SMPL model .pkl file or the official .npz file,
    # from which, a chumpy-free SMPL model .pkl file can be created.
    try:
        smpl_model = smplx.SMPL(
            model_path=smpl_model_file,
        )
    except Exception as e:
        print("Exception: ", e)
        print(
            "If the provided SMPL model file is a .pkl file, please make sure it is chumpy free."
        )
        if smpl_model_file.endswith(".npz"):
            converted_smpl_model_file = smpl_model_file.replace(
                ".npz", "_generate_from_npz.pkl"
            )
            if not os.path.exists(converted_smpl_model_file):
                smpl_model_data = dict(np.load(smpl_model_file))
                import pickle

                with open(converted_smpl_model_file, "wb") as f:
                    pickle.dump(smpl_model_data, f)
            smpl_model = smplx.SMPL(
                model_path=converted_smpl_model_file,
            )

os.makedirs(output_dir, exist_ok=True)

mhr_model = MHR.from_files(lod=1, device=device)
converter = Conversion(
    mhr_model=mhr_model, smpl_model=smpl_model, method="pymomentum"
)
example_output_dir = output_dir + "/smpl_para2mhr_pymomentum"
os.makedirs(example_output_dir, exist_ok=True)
    
"""Get SMPL(X) data for examples."""
phalp_data = joblib.load(input_phalp_poses_file)
num_frames = len(phalp_data.keys())

print("TOTAL FRAMES:", num_frames)

smpl_parameters = {}
# Prefer to get these from the PHALP output rather than fabricating them
#smpl_parameters["global_orient"] = np.zeros((num_frames, 3))
#smpl_parameters["betas"] = np.random.randn(num_frames, 10)
global_orients = []
betas = []
body_poses = []
for i, frame_image_id in enumerate(phalp_data):
    # This is where it could handle more than one pose per frame
    smpl_dict = phalp_data[frame_image_id]["smpl"][0]
    global_orient_back, _ = cv2.Rodrigues(smpl_dict["global_orient"][0])
    global_orient_back = global_orient_back.T
    flip_mat = np.diag(np.full(3, 1))
    flip_mat[2,2] = -1 # Need to flip the Z axis orientations
    global_orient_back = np.matmul(global_orient_back, flip_mat)
    global_orients.append(global_orient_back)
    betas.append(smpl_dict["betas"])

    body_rvecs = []

    for body_pose in smpl_dict["body_pose"]:
        rvec_back, _ = cv2.Rodrigues(body_pose)
        body_rvecs.append(rvec_back.T.tolist()[0])

    body_poses.append(body_rvecs)

    # What the heck is this
    #body_poses.append(np.concatenate([body_rvecs[3:66], np.zeros_like(body_rvecs[:6])], axis=-1))

smpl_parameters["betas"] = np.array(betas)
smpl_parameters["global_orient"] = np.array(global_orients)
smpl_parameters["body_pose"] = np.array(body_poses)

for k, v in smpl_parameters.items():
    smpl_parameters[k] = (
        torch.from_numpy(v).to(torch.float32).to(device)
    )
   
print("shape of global_orient", smpl_parameters["global_orient"].shape)
print("shape of body_pose", smpl_parameters["body_pose"].shape)
print("shape of betas", smpl_parameters["betas"].shape)

smpl_vertices = []
num_frames = smpl_parameters["body_pose"].shape[0]
for i in range(num_frames):
    frame_smpl_vertices = (
        smpl_model(
            global_orient=smpl_parameters["global_orient"][i : i + 1],
            body_pose=smpl_parameters["body_pose"][i : i + 1],
            betas=smpl_parameters["betas"][i : i + 1],
        )
        .vertices.detach()
        .cpu()
        .numpy()[0]
    )
    smpl_mesh = trimesh.Trimesh(
        frame_smpl_vertices, smpl_model.faces, process=False
    )
    smpl_mesh.export(f"{output_dir}/{i:03d}_smpl.ply")
    smpl_vertices.append(frame_smpl_vertices)

smpl_vertices = np.array(smpl_vertices)

print(
    "\nConverting SMPL to MHR with PyMomentum"
)

# It's better to run these as a big batch rather than one at a time, because 
#   PyMomentum is able to benefit from some tracking that it apparently does.
conversion_results = converter.convert_smpl2mhr(
    smpl_vertices=smpl_vertices,
    smpl_parameters=smpl_parameters,
    single_identity=True,
    return_mhr_meshes=True,
    return_mhr_vertices=True,
    return_mhr_parameters=True,
    return_fitting_errors=True,
)
print("Total conversions (including errors):", conversion_results.result_errors.shape)

for i, mesh in enumerate(conversion_results.result_meshes):
    # Save the results 
    #mesh.vertices /= 100.0 # This doesn't seem to do anything?
    #mesh.export(f"{example_output_dir}/{i:03d}_result_mhr.ply")
    mesh.export(f"{example_output_dir}/{i:03d}_result_mhr.obj")

mhr_vertices, skeleton_state = mhr_model(conversion_results.result_parameters["identity_coeffs"], conversion_results.result_parameters["lbs_model_params"], conversion_results.result_parameters["face_expr_coeffs"])

# mhr_vertices are 18439 3d-coords (18439 = LOD 1, which seems to be the default). Not included because this
#  data is redundant with conversion_results['result_vertices'] (and also probably 
#  conversion_results['result_meshes'])
# sekelton_state is 127 8-element vectors, so 8 for each joint. It's not obvious what these elements are.
# conversion_results includes result_parameters, which consists of
#   'lbs_model_params' # 204 per pose - 136 pose parameters + 68 skeletal transformation params
#   'identity_coeffs' # 45 per pose (body (20), head (20), hand (5) blendshapes)
#   'face_expr_coeffs' # 72 per pose
# it also has result_meshes, which is one Trimesh object per pose, which also has 18439 vertices
#   (can be exported as .obj and then converted to FBX via Blender, but without the armature)
# and it has result_vertices, which is a list of 18439 3d-coords, one per pose, identical to 
#   mhr_vertices, except with more significant digits

output_fn = os.path.basename(input_phalp_poses_file).replace(".pkl",".mhr.pkl")
joblib.dump({"skeleton_state": skeleton_state, "conversion_results": conversion_results}, f"{example_output_dir}/{output_fn}")
#joblib.dump({"mhr_vertices": mhr_vertices, "skeleton_state": skeleton_state, "conversion_results": conversion_results}, f"{example_output_dir}/{output_fn}")
