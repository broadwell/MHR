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

SMPL_MODEL_FILE = "/srv/mime/software/MHR/data/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"

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
# Can't we just get these from the PHALP output rather than fudging them?
#smpl_parameters["global_orient"] = np.zeros((num_frames, 3))
#smpl_parameters["betas"] = np.random.randn(num_frames, 10)
global_orients = []
betas = []
body_poses = []
for i, frame_image_id in enumerate(phalp_data):
    # This is where it could handle more than one pose per frame
    smpl_dict = phalp_data[frame_image_id]["smpl"][0]
    #smpl_pose = phalp_data[frame_image_id]["pose"][0] # PHALP needs to be run in visualization mode for this
    #global_orients.append(smpl_dict["global_orient"][0]) # can do [0] without losing anything....
    #print("shape of dict global_orient", smpl_dict["global_orient"][0].shape)
    global_orient_back, _ = cv2.Rodrigues(smpl_dict["global_orient"][0])
    #print("shape of post Rodrigues global_orient", global_orient_back.shape)
    #global_orients.append(np.zeros((3, 3)))
    global_orient_back = global_orient_back.T
    flip_mat = np.diag(np.full(3, 1))
    flip_mat[2,2] = -1
    global_orient_back = np.matmul(global_orient_back, flip_mat)
    global_orients.append(global_orient_back)
    #print("Shape of dict betas", smpl_dict["betas"].shape)
    betas.append(smpl_dict["betas"])
    #betas.append(np.random.randn(10))
    #body_poses.append(np.concatenate([smpl_pose[3:66], np.zeros_like(smpl_pose[:6])], axis=-1))
    #print("shape of body_pose", smpl_dict["body_pose"].shape)
    #print("shape of global_orient", smpl_dict["global_orient"].shape)
    #body_poses.append(smpl_dict["body_pose"])

    body_rvecs = []

    for body_pose in smpl_dict["body_pose"]:
        rvec_back, _ = cv2.Rodrigues(body_pose)
        #print(rvec_back.T.tolist()[0])
        body_rvecs.append(rvec_back.T.tolist()[0])

    body_poses.append(body_rvecs)

    # What the heck is this
    #body_poses.append(np.concatenate([body_rvecs[3:66], np.zeros_like(body_rvecs[:6])], axis=-1))
    #"body_pose": np.concatenate(
    #    [smplx_full_poses[:, 3:66], np.zeros_like(smplx_full_poses[:, :6])],
    #    axis=-1,
    #),
    #body_poses.append(smpl_dict["body_pose"].flatten()[:69])

smpl_parameters["betas"] = np.array(betas)
smpl_parameters["global_orient"] = np.array(global_orients)
smpl_parameters["body_pose"] = np.array(body_poses)

for k, v in smpl_parameters.items():
    smpl_parameters[k] = (
        torch.from_numpy(v).to(torch.float32).to(device)
    )
   
print("global_orient", smpl_parameters["global_orient"].shape)
print("body_pose", smpl_parameters["body_pose"].shape)
print("betas", smpl_parameters["betas"].shape)

num_frames = smpl_parameters["body_pose"].shape[0]
for i in range(num_frames):
    smplx_vertices = (
        smpl_model(
            global_orient=smpl_parameters["global_orient"][i : i + 1],
            body_pose=smpl_parameters["body_pose"][i : i + 1],
            betas=smpl_parameters["betas"][i : i + 1],
        )
        .vertices.detach()
        .cpu()
        .numpy()[0]
    )
    smplx_mesh = trimesh.Trimesh(
        smplx_vertices, smpl_model.faces, process=False
    )
    smplx_mesh.export(f"{output_dir}/{i:03d}_smpl.ply")

    print(
        "\nConverting SMPL parameters to MHR with PyMomentum"
    )

    try:
        conversion_results = converter.convert_smpl2mhr(
            smpl_vertices=smplx_vertices,
            smpl_parameters=smpl_parameters,
            single_identity=True,
            return_mhr_meshes=True,
            return_mhr_vertices=True,
            return_mhr_parameters=True,
            return_fitting_errors=True,
        )
        print("Conversion errors:")
        print(conversion_results.result_errors)

        mesh = conversion_results.result_meshes[0]
        mesh.vertices /= 100.0
    except Exception as e:
        print("Error in conversion:", e)

    # Save the results (or reuse the previous one if an error occurred)
    mesh.export(f"{example_output_dir}/{i:03d}_result_mhr.ply")

    print("Shape of results vertices", conversion_results.result_vertices.shape)
    print("Shape of lbs model params", conversion_results.result_parameters["lbs_model_params"].shape)
