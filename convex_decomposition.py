import pybullet as p
import argparse
import os


""" This file generates a folder called vhacd_meshes/ under the same parent folder as the input_dir """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    parent_dir = os.path.dirname(args.input_dir)
    meshes = os.listdir(args.input_dir)
    meshes.sort()
    output_dir = os.path.join(parent_dir, 'vhacd_meshes')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, m in enumerate(meshes):
        input_mesh_path = os.path.join(args.input_dir, m)
        output_mesh_path = os.path.join(output_dir, m)
        log_file_path = os.path.join(output_dir, f'{i}.txt')
        p.vhacd(fileNameIn=input_mesh_path,
                fileNameOut=output_mesh_path,
                fileNameLogging=log_file_path,
                resolution=16000000,
                depth=32)

