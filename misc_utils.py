import copy
import sys
from collections import OrderedDict
import os
import csv
import pybullet_data
import pybullet as p

import icp
import pybullet_utils as pu
import pandas as pd
from math import radians, cos, sin, sqrt, exp
import numpy as np
import matplotlib.pyplot as plt
import pybullet_utils as pu
import yaml
import json
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy.ndimage import rotate
import math
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# the definition of these directions are with respect to the occupancy grid
move_map = {
    0: 'up',
    1: 'right',
    2: 'down',
    3: 'left'
}

up = 0
right = 1
down = 2
left = 3

black = 0
white = 255
unexplored = 127
current_black = 63
current_white = 191


def show_img(img, title=None, ticks_off=False):
    """
    Universal show image function

    :param img: np.array,
        (height, width), uint8
        (1, height, width), uint8
        (height, width, 3), uint8
        (3, height, width), uint8
    :return:
    """
    img = img.astype(np.uint8)
    if img.ndim == 2:
        # (height, width)
        img = np.tile(img, (3, 1, 1))
        img = np.transpose(img, [1, 2, 0])
    elif img.ndim == 3:
        if img.shape[0] == 1:
            # (1, height, width)
            img = np.tile(img, (3, 1, 1))
            img = np.transpose(img, [1, 2, 0])
        elif img.shape[0] == 3:
            # (3, height, width)
            img = np.transpose(img, [1, 2, 0])
        else:
            pass
    show_rgb(img, title=title, ticks_off=ticks_off)


def save_img(img, path):
    """ Universal save image function """
    img = img.astype(np.uint8)
    if img.ndim == 2:
        # (height, width)
        img = np.tile(img, (3, 1, 1))
        img = np.transpose(img, [1, 2, 0])
    elif img.ndim == 3:
        if img.shape[0] == 1:
            # (1, height, width)
            img = np.tile(img, (3, 1, 1))
            img = np.transpose(img, [1, 2, 0])
        elif img.shape[0] == 3:
            # (3, height, width)
            img = np.transpose(img, [1, 2, 0])
        else:
            pass
    save_rgb(img, path)

def show_gray(img):
    """
    :param img: np.array, (height, width), uint8
    """
    # this map shows value relative
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()


def save_gray(img, dir, fnm):
    """
    :param img: np.array, (height, width), uint8
    """
    # this map shows value relative
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.imsave(os.path.join(dir, fnm), img, cmap=plt.get_cmap('gray'))
    plt.show()


def save_rgb(img, path):
    """
    :param img: np.array, (height, width, 3), uint8

    this plot and then save. so the image size can be much larger than the array size
    """
    dir, fnm = os.path.split(path)
    if dir != '' and not os.path.exists(dir):
        os.makedirs(dir)
    plt.imshow(img)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.tight_layout()
    plt.savefig(path, dpi=96, bbox_inches='tight', pad_inches=0)


def gray_2_binary(img, threshold=0):
    """ make pixel bigger than threshold white and pixel smaller than threshold black """
    img_ = copy.deepcopy(img)
    img_[img > threshold] = 255  # white
    img_[img <= threshold] = 0  # black
    return img_


def sample_number(num, arr_x, arr_y):
    """

    :param num:
    :param arr_x: (n, 28, 28)
    :param arr_y: (n, )
    :return:
    """
    indices = np.where(arr_y == num)
    index = np.random.choice(indices[0])
    return arr_x[index]


def save_yaml(data, path):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_json(data, path):
    # save checkpoint
    json.dump(data, open(path, 'w'), indent=4)


def convert_second(seconds):
    day = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02dd%02dh%02dm%02ds" % (day, hour, minutes, seconds)


def compute_next_loc(current_loc, move, height=28, width=28):
    """ return a tuple """
    # clockwise move
    if move == up:
        # move up
        h, w = up_in_grid(current_loc)
    elif move == right:
        # move right
        h, w = right_in_grid(current_loc, width)
    elif move == down:
        # move down
        h, w = down_in_grid(current_loc, height)
    elif move == left:
        # move left
        h, w = left_in_grid(current_loc)
    else:
        raise NotImplementedError('no such move!')
    return h, w


def find_neighbor(loc, height, width):
    h, w = loc
    neighbor_locs = []
    if h + 1 < height: neighbor_locs.append((h + 1, w))
    if h - 1 > -1: neighbor_locs.append((h - 1, w))
    if w + 1 < width: neighbor_locs.append((w + 1, w))
    if w - 1 > -1: neighbor_locs.append((w - 1, w))
    return neighbor_locs


def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file and write the first line if the file does not already exist """
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def sample_position(lower_limits=(0, 0, 0), upper_limits=(1, 1, 1)):
    """
    :param lower_limits: lower limits of (x, y, z)
    :param upper_limits: upper limits of (x, y, z)
    :return: a position between lower_limits and upper_limits
    """
    return np.random.uniform(low=lower_limits, high=upper_limits)


def show_rgb(img, title='rgb', ticks_off=False):
    """
    :param img: np.array, (height, width, 3), uint8
    """
    # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('rgb image', RGB_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # plt.imshow(img)
    # return RGB_img
    if title is not None:
        plt.title(title)
    plt.imshow(img)
    if ticks_off:
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)  # labels along the bottom edge are off
    plt.tight_layout()
    plt.show()


def show_depth(img, title='depth', ticks_off=False):
    """
    :param img: np.array, (height, width), float32, processed distance in the range of [0, 1]
    """
    # plt.figure()  # if you want a separate figure window
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.tight_layout()
    if ticks_off:
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)  # labels along the bottom edge are off
    plt.pause(0.0001)


def show_rgbs(imgs, title='rgb'):
    """
    :param imgs: np.array, (n, 3, height, width), uint8
    """
    for rgb in imgs:
        rgb = np.transpose(rgb, (1, 2, 0))
        show_rgb(rgb)


def show_depths(imgs, title='depth'):
    """
    :param imgs: np.array, (n, 1, height, width), float32, processed distance in the range of [0, 1]
    """
    # plt.figure()  # if you want a separate figure window
    for depth in imgs:
        depth = np.squeeze(depth)
        show_depth(depth)


def show_segmentation(img):
    """
    :param img: np.array, (height, width), int32, each pixel corresponds to an object id
    """
    # plt.figure()  # if you want a separate figure window
    plt.imshow(img)
    plt.tight_layout()
    plt.pause(0.0001)


def draw_workspace(lower_limits, upper_limits, rgb_color=(0, 1, 0)):
    markers = []
    lines = [((lower_limits[0], lower_limits[1], lower_limits[2]), (lower_limits[0], upper_limits[1], lower_limits[2])),
             ((lower_limits[0], lower_limits[1], lower_limits[2]), (upper_limits[0], lower_limits[1], lower_limits[2])),
             ((lower_limits[0], upper_limits[1], lower_limits[2]), (upper_limits[0], upper_limits[1], lower_limits[2])),
             ((upper_limits[0], lower_limits[1], lower_limits[2]), (upper_limits[0], upper_limits[1], lower_limits[2])),

             ((lower_limits[0], lower_limits[1], upper_limits[2]), (lower_limits[0], upper_limits[1], upper_limits[2])),
             ((lower_limits[0], lower_limits[1], upper_limits[2]), (upper_limits[0], lower_limits[1], upper_limits[2])),
             ((lower_limits[0], upper_limits[1], upper_limits[2]), (upper_limits[0], upper_limits[1], upper_limits[2])),
             ((upper_limits[0], lower_limits[1], upper_limits[2]), (upper_limits[0], upper_limits[1], upper_limits[2])),

             ((lower_limits[0], lower_limits[1], lower_limits[2]), (lower_limits[0], lower_limits[1], upper_limits[2])),
             ((lower_limits[0], upper_limits[1], lower_limits[2]), (lower_limits[0], upper_limits[1], upper_limits[2])),
             ((upper_limits[0], lower_limits[1], lower_limits[2]), (upper_limits[0], lower_limits[1], upper_limits[2])),
             ((upper_limits[0], upper_limits[1], lower_limits[2]), (upper_limits[0], upper_limits[1], upper_limits[2]))]

    for start_pos, end_pos in lines:
        markers.append(pu.draw_line(start_pos, end_pos, rgb_color))
    return markers


def draw_workspace_xy(lower_limits, upper_limits, z, rgb_color=(0, 1, 0)):
    markers = []
    lines = [((lower_limits[0], lower_limits[1], z), (lower_limits[0], upper_limits[1], z)),
             ((lower_limits[0], lower_limits[1], z), (upper_limits[0], lower_limits[1], z)),
             ((lower_limits[0], upper_limits[1], z), (upper_limits[0], upper_limits[1], z)),
             ((upper_limits[0], lower_limits[1], z), (upper_limits[0], upper_limits[1], z))]

    for start_pos, end_pos in lines:
        markers.append(pu.draw_line(start_pos, end_pos, rgb_color))
    return markers


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def create_object_urdf(object_mesh_filepath, object_name,
                       urdf_template_filepath='assets/object_template.urdf',
                       urdf_target_object_filepath='assets/target_object.urdf'):
    # set_up urdf
    os.system('cp {} {}'.format(urdf_template_filepath, urdf_target_object_filepath))
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_mesh_filepath', object_mesh_filepath, urdf_target_object_filepath)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name', object_name, urdf_target_object_filepath)
    os.system(sed_cmd)
    return urdf_target_object_filepath


def create_extruded_polygon(polygon_id):
    mesh_folder = 'assets/objects_urdf/extruded_polygons/meshes'
    urdf_folder = 'assets/objects_urdf/extruded_polygons/urdf'
    urdf_target_object_filepath = os.path.join(urdf_folder, f'{polygon_id}.urdf')
    mesh_filepath = os.path.join(mesh_folder, f'{polygon_id}.stl')
    if not os.path.exists(urdf_folder):
        os.makedirs(urdf_folder)
    if not os.path.exists(urdf_target_object_filepath):
        urdf_target_object_filepath = create_object_urdf(mesh_filepath,
                                                         str(polygon_id),
                                                         urdf_template_filepath='assets/objects_urdf/extruded_polygons/object_template.urdf',
                                                         urdf_target_object_filepath=urdf_target_object_filepath)
    return urdf_target_object_filepath


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w


def check_class_count_oris(partial_grid, gt_grids):
    """
    partial_grid: (1, height, width)
    gt_grids: (num_classes, num_rotations, height, width)
    """
    # TODO verify this works with one random orientation
    # will return all 0s for porbs if not matching anything

    # I dont want to change any of them
    gt_grids_cp = copy.deepcopy(gt_grids)

    # duplicate the partial grid to be the same shape of gt_grids
    partial_grids = np.tile(partial_grid, gt_grids.shape[:2] + (1, 1))
    occluded_indices = np.where(partial_grids == unexplored)
    gt_grids_cp[occluded_indices] = unexplored
    # (num_classes, num_rotations)
    equals = np.all(gt_grids_cp == partial_grids, axis=(2, 3))
    # assert any(equals), "partial map should match at least one ground truth"
    # (num_classes, )
    num_equals = np.sum(equals, axis=-1)
    probs = num_equals / np.sum(num_equals)
    # print(probs)
    prediction = np.argmax(probs)
    max_prob = np.max(probs)
    return prediction, max_prob, probs


def check_class_bkup(partial_grid, gt_grids):
    """
    partial_grid: (1, height, width)
    gt_grids: (num_classes, num_rotations, height, width)
    """
    # TODO verify this works with one random orientation
    # will return all 0s for porbs if not matching anything

    # I dont want to change any of them
    gt_grids_cp = copy.deepcopy(gt_grids)

    # duplicate the partial grid to be the same shape of gt_grids
    partial_grids = np.tile(partial_grid, gt_grids.shape[:2] + (1, 1))
    occluded_indices = np.where(partial_grids == black)
    gt_grids_cp[occluded_indices] = black
    # (num_classes, num_rotations)
    equals = np.all(gt_grids_cp == partial_grids, axis=(2, 3))
    # (num_classes, )
    equals = np.any(equals, axis=-1)
    # assert any(equals), "partial map should match at least one ground truth"
    num_equals = np.count_nonzero(equals)
    probs = [1 / num_equals if equal else 0 for equal in equals]
    prediction = np.argmax(probs)
    max_prob = np.max(probs)
    return prediction, max_prob, probs


def check_class(partial_grid, gt_grids):
    """
    partial_grid: (1, height, width)
    gt_grids: (num_classes, num_rotations, height, width)
    """
    # TODO verify this works with one random orientation
    # will return all 0s for porbs if not matching anything

    # I dont want to change any of them
    gt_grids_cp = copy.deepcopy(gt_grids)

    # duplicate the partial grid to be the same shape of gt_grids
    partial_grids = np.tile(partial_grid, gt_grids.shape[:2] + (1, 1))
    occluded_indices = np.where(partial_grids == unexplored)
    gt_grids_cp[occluded_indices] = unexplored
    # (num_classes, num_rotations)
    equals = np.all(gt_grids_cp == partial_grids, axis=(2, 3))
    # (num_classes, )
    equals = np.any(equals, axis=-1)
    # assert any(equals), "partial map should match at least one ground truth"
    num_equals = np.count_nonzero(equals)
    probs = [1 / num_equals if equal else 0 for equal in equals]
    prediction = np.argmax(probs)
    max_prob = np.max(probs)
    return prediction, max_prob, probs


def get_current_loc(ob):
    """ ob is (1, height, width). (height, width) will not work """
    x_idx, y_idx = np.where(np.logical_or(ob[0] == current_black, ob[0] == current_white))
    assert len(x_idx) == 1 and len(y_idx) == 1
    x_idx, y_idx = x_idx[0], y_idx[0]
    return x_idx, y_idx


def get_discriminator_input(obs):
    if obs.ndim == 3:
        # (1, height, width)
        current_loc = get_current_loc(obs)
        # discriminator does not care about agent location
        discriminator_input = copy.deepcopy(obs)
        discriminator_input[0][current_loc] = black \
            if discriminator_input[0][current_loc] == current_black else white
    elif obs.ndim == 4:
        # (n, 1, height, width)
        n = obs.shape[0]
        discriminator_input = [get_discriminator_input(obs[i]) for i in range(n)]
        discriminator_input = np.stack(discriminator_input, axis=0)
    else:
        raise TypeError
    return discriminator_input


def find_not_go_back_moves(ob):
    # ob is of shape (1, height, width)
    # return neighbor pixels that are not explored
    # if all are explored, return []
    height, width = ob[0].shape
    current_loc = get_current_loc(ob)
    good_moves = []

    for a in range(4):
        next_loc = compute_next_loc(current_loc, a, height, width)
        if ob[0][next_loc] == unexplored:
            good_moves.append(a)
    return good_moves


def assemble_action(move, done, prediction):
    return {'move': move, 'done': done, 'prediction': prediction}


def construct_discriminator(discriminator_type, height, width, discriminator_path=None, num_models=5, lr=0.001, dataset='extruded_polygons_r_0.1_s_8_h_0.05'):
    from discriminator import GroundTruthDiscriminator, DummyDiscriminator, LearnedDiscriminator, EnsembleDiscriminator, ICPDiscriminator
    if discriminator_type == 'gt':
        discriminator = GroundTruthDiscriminator(discriminator_path)
    elif discriminator_type == 'dummy':
        discriminator = DummyDiscriminator(num_classes=10)
    elif discriminator_type == 'learned':
        discriminator = LearnedDiscriminator(height=height,
                                             width=width,
                                             model_path=discriminator_path,
                                             lr=lr)
    elif discriminator_type == 'ensemble':
        discriminator = EnsembleDiscriminator(num_models=num_models,
                                              height=height,
                                              width=width,
                                              model_path=discriminator_path)
    elif discriminator_type == 'icp':
        discriminator = ICPDiscriminator(dataset=dataset)
    else:
        raise TypeError
    return discriminator


def construct_explorer(explorer_type, image_size, explorer_path, discriminator=None):
    from explorer import RandomExplorer, PPOExplorer, EdgeFollowExplorer, InfoGainExplorer, NotGoBackExplorer, AllInONeExplorer
    if explorer_type == 'random':
        e = RandomExplorer(4)
    elif explorer_type == 'ppo':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        e = PPOExplorer(action_dim=4, device=device, model_path=explorer_path, img_size=image_size)
    elif explorer_type == 'edge':
        e = EdgeFollowExplorer(image_size)
    elif explorer_type == 'info':
        e = InfoGainExplorer(discriminator)
    elif explorer_type == 'not_go_back':
        e = NotGoBackExplorer()
    elif explorer_type == 'all_in_one':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        e = AllInONeExplorer(action_dim=14, device=device, model_path=explorer_path, img_size=image_size)
    else:
        raise TypeError
    return e


def get_action(explorer, discriminator, obs, gt_discriminator=None, terminal_confidence=1.0):
    action = {}

    if discriminator is not None:
        prediction, max_prob, probs = discriminator.predict(obs)

    if gt_discriminator is not None:
        # gt discriminator is added here for comparison
        gt_prediction, gt_max_prob, gt_probs = gt_discriminator.predict(obs)

    if explorer.pattern == 'all_in_one':
        move = explorer.get_move(obs)
        if 0 <= move < 4:
            action['move'] = move
            action['prediction'] = 0
            action['max_prob'] = 0.1
            action['probs'] = np.full(10, 0.1)
            action['done'] = False
        else:
            action['move'] = 0
            action['prediction'] = move - 4
            action['probs'] = np.zeros(10)
            action['probs'][action['prediction']] = 1
            action['done'] = True
            action['max_prob'] = 1
    else:
        action['move'] = explorer.get_move(obs)
        action['prediction'] = prediction
        action['done'] = max_prob >= terminal_confidence
        action['max_prob'] = max_prob
        action['probs'] = probs
    return action


def clip(x, a=0, b=1):
    assert a < b
    x = a if x < a else x
    x = b if x > b else x
    return x


def sparse(a=0, b=1):
    def f(x):
        x = clip(x, a, b)
        if x == b:
            y = 1
        else:
            y = 0
        return y
    return f


def dense_quart_circle():
    # deprecated
    def f(x):
        y = - sqrt(1 - x * x) + 1
        return y

    return f


def dense_linear(a=0, b=1):
    def f(x):
        x = clip(x, a, b)
        return (x - a) / (b - a)
    return f


def dense_exp(knob, a=0, b=1):
    def f(x):
        x = clip(x, a, b)
        y = (exp(knob * x) - exp(knob * a)) / (exp(knob * b) - exp(knob * a))
        return y

    return f


def get_prob_mapping_function(reward_type, a=0, b=1, knob=None):
    if reward_type == 'sparse':
        return sparse(a, b)
    elif reward_type == 'dense_linear':
        return dense_linear(a, b)
    elif reward_type == 'dense_exp':
        return dense_exp(knob, a, b)
    else:
        raise TypeError('not recognized reward type')


def rotate_img(img, angle):
    """

    :param img: (height, width) numpy array
    :param angle: in degrees
    :return: (height, width) numpy array
    """
    return rotate(img, angle, order=0, reshape=False, cval=127)


def rotate_imgs(imgs, angles):
    """

    :param imgs: (n, 1, height, width) numpy array
    :param angles: list, in degrees
    :return: (n, 1, height, width) numpy array
    """
    imgs_copy = copy.deepcopy(imgs)
    for i in range(imgs_copy.shape[0]):
        imgs_copy[i][0] = rotate_img(imgs[i][0], angles[i])
    return imgs_copy


def generate_rotated_imgs(img, num_rotations):
    """

    :param img: (1, height, width) numpy array
    :param num_rotations: int
    :return: (num_rotations, 1, height, width) numpy array
    """
    if num_rotations == 1:
        return img[None, ...]
    else:
        gap = round(360 / num_rotations)
        angles = [0 + gap * i for i in range(num_rotations)]
        imgs = [rotate_img(img[0], a)[None, ...] for a in angles]
        imgs = np.stack(imgs)
        return imgs


def compute_class_balance(X, Y):
    assert X.shape[0] == Y.shape[0]
    classes, freq = np.unique(Y, return_counts=True)
    percents = freq / np.sum(freq)
    stats = {c: p for c, p in zip(classes, percents)}
    return stats


def compute_pixel_freq(X, visualize=False, save=False, save_path=None):
    """ compute the frequency of num of explored pixels """
    num_explored_pixels = np.count_nonzero(X != unexplored, axis=(1, 2, 3))
    pixels, freq = np.unique(num_explored_pixels, return_counts=True)

    # using historgram
    # range = (min(num_explored_pixels), max(num_explored_pixels))
    # bins = math.ceil(range[1] - range[0] / 5)
    # plt.hist(num_explored_pixels, range=range, bins=bins)

    if visualize or save:
        plt.bar(pixels, height=freq, width=0.8)
        if save:
            head, tail = os.path.split(save_path)
            if not os.path.exists(head):
                os.makedirs(head)
            plt.savefig(save_path)
        if visualize:
            plt.show()
        plt.clf()
    stats = {p: f for p, f in zip(pixels, freq)}
    return stats


def load_dataset(dataset_dir):
    grids = np.load(os.path.join(dataset_dir, 'grids.npy'))
    grids_border = np.load(os.path.join(dataset_dir, 'grids_border.npy'))
    rgb = np.load(os.path.join(dataset_dir, 'rgb.npy'))
    border_neighbors_arr = np.load(os.path.join(dataset_dir, 'border_neighbors_arr.npy'), allow_pickle=True)
    return grids, grids_border, rgb, border_neighbors_arr


def seed_env(env, seed):
    # this is important for reproducibility
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def construct_loaders(dataset, split=0.2, seed=10, train_batch_size=64, test_batch_size=1000):
    dataset_size = len(dataset)
    test_size = int(np.floor(split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True)
    return train_loader, test_loader


def save_command(file_path):
    args = copy.deepcopy(sys.argv)
    args[0] = os.path.basename(args[0])
    command = ' '.join(['python'] + args)
    print(command)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    text_file = open(file_path, "w")
    text_file.write(command)
    text_file.close()


def degree_2_radian(d):
    return [math.radians(i) for i in d]


def radian_2_degree(r):
    return [math.degrees(i) for i in r]


def get_neighbour_locs(loc, height, width):
    """
    returns [(up), (right), (down), (left)]
    will return None if not exists (wall)
    """
    neighbours = []
    for i in range(4):
        v = compute_next_loc(loc, i, height, width)
        if v == loc:
            neighbours.append(None)
        else:
            neighbours.append(v)
    return neighbours


def up_in_grid(loc):
    return loc[0] - 1 if loc[0] - 1 >= 0 else 0, loc[1]


def right_in_grid(loc, right_limit):
    return loc[0], loc[1] + 1 if loc[1] + 1 < right_limit else right_limit - 1


def down_in_grid(loc, down_limit):
    return loc[0] + 1 if loc[0] + 1 < down_limit else down_limit - 1, loc[1]


def left_in_grid(loc):
    return loc[0], loc[1] - 1 if loc[1] - 1 >= 0 else 0


def get_direction(loc1, loc2):
    """ Get the direction from loc1 to loc2, loc1 and loc2 are not the same but neighboring """
    assert loc1 != loc2
    direction = None
    if loc2[0] == loc1[0] + 1:
        direction = down
    if loc2[0] == loc1[0] - 1:
        direction = up
    if loc2[1] == loc1[1] + 1:
        direction = right
    if loc2[1] == loc1[0] - 1:
        direction = left
    return direction


def get_next_direction_clockwise(direction):
    if direction == up:
        return right
    if direction == right:
        return down
    if direction == down:
        return left
    if direction == left:
        return up


def get_next_direction_anti_clockwise(direction):
    if direction == up:
        return left
    if direction == right:
        return up
    if direction == down:
        return right
    if direction == left:
        return down


def check_grid_collision(old_ob, new_ob):
    """
    Compare two observations and check if there is a new white pixel explored, return the location of that white pixel
    old_ob and new_ob are (1, height, width)
    """
    dim_0, dim_1, dim_2 = np.where(old_ob != new_ob)
    if len(dim_0) == 1:
        # there is only one location that is different
        loc = (dim_1[0], dim_2[0])
        if new_ob[0][loc] == white and old_ob[0][loc] == unexplored:
            return True, loc
    return False, None


def convert_grid_2_pc(obs, step_size=0.005):
    """
    obs is a numpy array of shape (height, width), (1, height, width) or (n, 1, height, shape)

    return a numpy array of shape (m, 2) or (n, m, 2)
    """
    if obs.ndim == 2:
        dim0, dim1 = np.where(obs == white)
        points = [[x * step_size, y * step_size] for (x, y) in zip(dim0, dim1)]
        return np.array(points)
    elif obs.ndim == 3:
        dim0, dim1, dim2 = np.where(obs == white)
        points = [[x * step_size, y * step_size] for (x, y) in zip(dim1, dim2)]
        return np.array(points)
    else:
        raise NotImplementedError


def visualize_icp(src, dst, T):
    """ Visualize the quality of the computed transform """
    src_ = np.ones((src.shape[0], 3))
    src_[:, :2] = src
    dst_prime = np.dot(T, src_.T).T
    plt.scatter(dst[:, 0], dst[:, 1], label='dst')
    plt.scatter(src[:, 0], src[:, 1], label='src')
    plt.scatter(dst_prime[:, 0], dst_prime[:, 1], label='dst_prime')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def rotate_along_point(theta, x, y):
    # compute the homogeneous transformation matrix to rotate point along an arbitrary point
    # https://math.stackexchange.com/questions/2093314/rotation-matrix-of-rotation-around-a-point-other-than-the-origin
    return np.array([
        [np.cos(theta), -np.sin(theta), -x*np.cos(theta) + y*np.sin(theta) + x],
        [np.sin(theta), np.cos(theta), -x*np.sin(theta) - y*np.cos(theta) + y],
        [0, 0, 1]]
    )


def icp_with_random_init_ori(src, dst, num_ori):
    angles = np.linspace(0, 360, num_ori, endpoint=False)
    errors = []
    iters = []
    Ts = []
    for a in angles:
        theta = math.radians(a)
        init_pose = rotate_along_point(theta, 0.15, 0.15)
        T, distances, i = icp.icp(src, dst, init_pose=init_pose, max_iterations=1000, tolerance=10e-7)
        error = np.mean(distances)
        Ts.append(T)
        errors.append(error)
        iters.append(i)
    min_error = np.min(errors)
    angle = angles[np.argmin(errors)]
    iter = iters[np.argmin(errors)]
    T = Ts[np.argmin(errors)]
    return T, min_error, iter, angle


def expand_occupancy_grid(g, times=5):
    """
    g: (h, w) or (1, h, w) numpy array
    return (h, w, 3) numpy array
    """
    if g.ndim == 2:
        g = np.repeat(g, times, axis=0)
        g = np.repeat(g, times, axis=1)
        g = np.tile(g, (3, 1, 1))
        g = np.transpose(g, [1, 2, 0])
    elif g.ndim == 3:
        g = np.repeat(g, times, axis=1)
        g = np.repeat(g, times, axis=2)
        g = np.tile(g, (3, 1, 1))
        g = np.transpose(g, [1, 2, 0])
    else:
        raise TypeError
    return g