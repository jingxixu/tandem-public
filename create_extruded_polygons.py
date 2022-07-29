import trimesh
import trimesh.path.polygons as tp
import trimesh.creation as tc
from trimesh.exchange.obj import export_obj
import numpy as np
import os
import argparse
import misc_utils as mu
import pybullet as p


"""
This file creates a folder
    - meshes
    - concave_meshes
    - info 
    - urdfs 
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--radius', type=float, default=0.1,
                        help='the approximate radius of the random 2D polygons to extrude from.')
    parser.add_argument('--height', type=float, default=0.05,
                        help='the height of the extruded polygons.')
    parser.add_argument('--max_sides', type=int, default=8,
                        help='maximum number of sides the random 2D polygon can have.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    p.connect(p.DIRECT)
    info_dir = os.path.join(args.save_dir, 'info')
    mesh_dir = os.path.join(args.save_dir, 'meshes')
    urdf_dir = os.path.join(args.save_dir, 'urdfs')
    concave_mesh_dir = os.path.join(args.save_dir, 'concave_meshes')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    if not os.path.exists(urdf_dir):
        os.makedirs(urdf_dir)
    if not os.path.exists(concave_mesh_dir):
        os.makedirs(concave_mesh_dir)
    np.random.seed(10)

    # example, clockwise
    polygon_vertices = [
        [0, 0],
        [0, 1],
        [2, 2],
        [1, 3],
        [3, 4],
        [4, 3],
        [3, 1]]
    polygon = tp.paths_to_polygons([polygon_vertices])[0]

    # sampling random polygon and extrude to 3D mesh, and create meshes
    for i in range(10):
        polygon = tp.random_polygon(segments=args.max_sides, radius=args.radius)
        tp.plot(polygon, show=False)
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(info_dir, f'{i}_polygon.png'))
        plt.clf()
        mesh = tc.extrude_polygon(polygon, height=args.height)
        # mesh.show()
        mesh.export(os.path.join(mesh_dir, f'{i}.obj'))

    # create concave obj
    for i in range(10):
        source_obj_file = os.path.join(mesh_dir, f'{i}.obj')
        dest_obj_file = os.path.join(concave_mesh_dir, f'{i}.obj')
        log_file = os.path.join(concave_mesh_dir, f'{i}.txt')
        p.vhacd(source_obj_file, dest_obj_file, log_file)

    # create urdfs
    for i in range(10):
        urdf_filepath = os.path.join(urdf_dir, f'{i}.urdf')
        mesh_filepath = os.path.join(os.path.join(concave_mesh_dir, f'{i}.obj'))
        urdf_filepath = mu.create_object_urdf(mesh_filepath,
                                              str(i),
                                              urdf_target_object_filepath=urdf_filepath)
