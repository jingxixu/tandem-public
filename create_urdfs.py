import pybullet as p
import argparse
import os
import misc_utils as mu


""" This file generates a folder called output_dir under the same parent folder as the input_dir """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    parent_dir = os.path.dirname(args.input_dir)
    meshes = os.listdir(args.input_dir)
    meshes = [m for m in meshes if m.endswith('.obj')]
    meshes.sort()
    output_dir = os.path.join(parent_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, m in enumerate(meshes):
        input_mesh_path = os.path.join(args.input_dir, m)
        urdf_filepath = os.path.join(output_dir, f'{i}.urdf')
        mesh_filepath = os.path.join(args.input_dir, m)
        urdf_filepath = mu.create_object_urdf(mesh_filepath,
                                              str(i),
                                              urdf_target_object_filepath=urdf_filepath)


