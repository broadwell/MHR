"""
pixi run python pkl_to_fbx.py path/to/mhr/output.pkl
"""
import argparse
import os
import sys

import joblib
import torch

sys.path.append("/srv/mime/software/momentum")

from mhr.mhr import MHR

_OUTPUT_DIR = "./tmp_results"  # Directory to store conversion results

if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mhr_model = MHR.from_files(lod=1, device=device)

    input_pkl_file = sys.argv[1]
    output_dir = _OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    mhr_data = joblib.load(input_pkl_file)

    bn = os.path.basename(input_pkl_file).replace(".mhr.pkl", "")

    glb_file = f"{output_dir}/{bn}_animation.glb"
    try:
        mhr_model.save_to_gltf(
            glb_file,
            mhr_data["conversion_results"].result_parameters["identity_coeffs"],
            mhr_data["conversion_results"].result_parameters["lbs_model_params"],
            mhr_data["conversion_results"].result_parameters["face_expr_coeffs"],
            fps=30,
        )
        print(f"Saved animation to {glb_file}")
    except Exception as e:
        print(f"An error occurred when exporting to gltf: {e}")

    fbx_file = f"{output_dir}/{bn}_animation.fbx"
    try:
        mhr_model.save_to_fbx(
            fbx_file,
            mhr_data["conversion_results"].result_parameters["identity_coeffs"],
            mhr_data["conversion_results"].result_parameters["lbs_model_params"],
            mhr_data["conversion_results"].result_parameters["face_expr_coeffs"],
            fps=30,
        )
        print(f"Saved animation to {fbx_file}")
    except Exception as e:
        print(f"An error occurred when exporting to fbx: {e}")
