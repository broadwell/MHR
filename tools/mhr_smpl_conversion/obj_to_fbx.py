"""
pixi run python mhr_to_fbx.py path/to/mhr/output.pkl
"""
import argparse
import os
import sys

import subprocess

_OUTPUT_DIR = "./tmp_results"  # Directory to store conversion results

def convert_to_fbx(input_path, output_path):
    # This command tells Blender to run a one-line Python script in the background
    command = [
        'blender', '--background', '--python-expr',
        f"import bpy; bpy.ops.wm.obj_import(filepath='{input_path}'); bpy.ops.export_scene.fbx(filepath='{output_path}')"
    ]
    subprocess.run(command)

if __name__ == "__main__":

    input_obj_file = sys.argv[1]
    output_dir = _OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    outfile = f"{output_dir}/{os.path.basename(input_obj_file).replace('.obj', '.fbx')}"

    convert_to_fbx(input_obj_file, outfile)
