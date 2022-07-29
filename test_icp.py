import copy
import numpy as np
import time
import icp
import matplotlib.pyplot as plt
import math
np.set_printoptions(suppress=True)
import misc_utils as mu

# Constants
sample_size = None


if __name__ == "__main__":
    # load point clouds
    pc = np.load('assets/datasets/extruded_polygons_r_0.1_s_8_h_0.05/point_clouds/point_clouds.npy', allow_pickle=True)
    np.random.seed(10)

    for dst in pc:
        # dst is the original point cloud we are trying to match
        trans = np.random.uniform(-0.01, 0.01, size=2)
        theta = -np.pi/4
        print(f'trans: {trans}')
        print(f'angle: {math.degrees(theta)}')
        # translation matrix
        trans_m = np.array([
            [1, 0, trans[0]],
            [0, 1, trans[1]],
            [0, 0, 1]]
        )
        # rotation matrix
        rot_m = mu.rotate_along_point(theta, 0.15, 0.15)
        # homogeneous transformation matrix
        homo_m = trans_m.dot(rot_m)

        # compute src point cloud
        dst_ = np.ones((dst.shape[0], 3))
        dst_[:, :2] = dst
        src_ = homo_m.dot(dst_.T).T
        src = src_[:, :2]

        # test transform
        if sample_size is not None:
            if sample_size < src.shape[0]:
                indices = np.random.choice(range(src.shape[0]), sample_size, replace=False)
                src = src[indices]
        # T, distances, i = icp.icp(src, dst, max_iterations=1000, tolerance=0.0000001)
        T, error, i, angle = mu.icp_with_random_init_ori(src, dst, num_ori=36)
        dst_prime = np.dot(T, src_.T).T
        print(f'error: {error}, iter: {i}, angle: {angle}')

        plt.scatter(dst[:, 0], dst[:, 1], label='dst')
        plt.scatter(src[:, 0], src[:, 1], label='src')
        plt.scatter(dst_prime[:, 0], dst_prime[:, 1], label='dst_prime')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.legend()
        plt.tight_layout()
        plt.show()
