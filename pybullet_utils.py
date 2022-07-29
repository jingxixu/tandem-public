from collections import namedtuple
import pybullet as p
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict, deque, namedtuple
from itertools import product, combinations, count
from transformations import quaternion_from_matrix
import pybullet_data

INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI


def configure_pybullet(rendering=False, debug=False, yaw=50.0, pitch=-35.0, dist=1.2, target=(0.0, 0.0, 0.0)):
    if not rendering:
        client = p.connect(p.DIRECT)
    else:
        client = p.connect(p.GUI)  # be careful about GUI vs GUI_SERVER, GUI_SERVER should only be used with shared memory
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    return client


def step(duration=1.0, client=0):
    for i in range(int(duration * 240)):
        p.stepSimulation(physicsClientId=client)


def step_real(duration=1.0, client=0):
    for i in range(int(duration * 240)):
        p.stepSimulation(physicsClientId=client)
        time.sleep(1.0 / 240.0)


def split_7d(pose):
    return [list(pose[:3]), list(pose[3:])]


def merge_pose_2d(pose):
    return pose[0] + pose[1]


def euler_from_quaternion(quaternion):
    return list(p.getEulerFromQuaternion(quaternion))


def quaternion_from_euler(euler):
    return list(p.getQuaternionFromEuler(euler))


def change_quat_rep(quaternion):
    """
    change the representation of quaternion from [w, x, y, z] to [x, y, z, w].
    ROS and pybullet are using [x, y, z, w], transformations.py is using [w, x, y, z]
    """
    quaternion = list(quaternion)
    return quaternion[1:] + [quaternion[0]]


def matrix_from_pose(pose_2d, client=0):
    (point, quat) = pose_2d
    matrix = np.eye(4)
    matrix[:3,3] = point
    matrix[:3,:3] = np.array(p.getMatrixFromQuaternion(quat, physicsClientId=client)).reshape(3, 3)
    return matrix


def pose_from_matrix(matrix, client=0):
    return [list(matrix[:3, 3]), change_quat_rep(quaternion_from_matrix(matrix))]


# Constraints

ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild',
                                               'maxAppliedForce'])


def remove_all_constraints(client=0):
    for cid in get_constraint_ids(client):
        p.removeConstraint(cid, physicsClientId=client)


def get_constraint_ids(client=0):
    """
    getConstraintUniqueId will take a serial index in range 0..getNumConstraints,  and reports the constraint unique id.
    Note that the constraint unique ids may not be contiguous, since you may remove constraints.
    """
    return sorted([p.getConstraintUniqueId(i, physicsClientId=client) for i in range(p.getNumConstraints(physicsClientId=client))])


def get_constraint_info(constraint, client):
    # there are four additional arguments
    return ConstraintInfo(*p.getConstraintInfo(constraint, physicsClientId=client)[:11])


# Joints

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute',  # 0
    p.JOINT_PRISMATIC: 'prismatic',  # 1
    p.JOINT_SPHERICAL: 'spherical',  # 2
    p.JOINT_PLANAR: 'planar',  # 3
    p.JOINT_FIXED: 'fixed',  # 4
    p.JOINT_POINT2POINT: 'point2point',  # 5
    p.JOINT_GEAR: 'gear',  # 6
}


def get_num_joints(body, client=0):
    return p.getNumJoints(body, physicsClientId=client)


def get_joints(body, client=0):
    return list(range(get_num_joints(body, client)))


def get_joint(body, joint_or_name):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name)
    return joint_or_name


JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])


def get_joint_info(body, joint, client=0):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=client))


def get_joints_info(body, joints):
    return [JointInfo(*p.getJointInfo(body, joint)) for joint in joints]


def get_joint_name(body, joint, client=0):
    return get_joint_info(body, joint, client).jointName.decode('UTF-8')


def get_joint_names(body):
    return [get_joint_name(body, joint) for joint in get_joints(body)]


def joint_from_name(body, name):
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return joint
    raise ValueError(body, name)


def has_joint(body, name):
    try:
        joint_from_name(body, name)
    except ValueError:
        return False
    return True


def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)


JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])


def get_joint_state(body, joint, client=0):
    return JointState(*p.getJointState(body, joint, physicsClientId=client))


def get_joint_position(body, joint, client=0):
    return get_joint_state(body, joint, client).jointPosition


def get_joint_torque(body, joint, client=0):
    return get_joint_state(body, joint, client).appliedJointMotorTorque


def get_joint_positions(body, joints=None, client=0):
    return list(get_joint_position(body, joint, client) for joint in joints)


def set_joint_position(body, joint, value, client=0):
    p.resetJointState(body, joint, value, physicsClientId=client)


def set_joint_positions(body, joints, values, client=0):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value, client)


def get_configuration(body, client=0):
    return get_joint_positions(body, get_movable_joints(body, client), client)


def set_configuration(body, values, client=0):
    set_joint_positions(body, get_movable_joints(body), values, client)


def get_full_configuration(body, client=0):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body), client)


def get_joint_type(body, joint, client=0):
    return get_joint_info(body, joint, client).jointType


def is_movable(body, joint, client=0):
    return get_joint_type(body, joint, client) != p.JOINT_FIXED


def get_movable_joints(body, client=0):  # 45 / 87 on pr2
    return [joint for joint in get_joints(body, client) if is_movable(body, joint)]


def joint_from_movable(body, index, client=0):
    return get_joints(body, client)[index]


def is_circular(body, joint, client=0):
    joint_info = get_joint_info(body, joint, client)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit


def get_joint_limits(body, joint, client=0):
    """
    Obtain the limits of a single joint
    :param body: int
    :param joint: int
    :return: (int, int), lower limit and upper limit
    """
    if is_circular(body, joint, client):
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint, client)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit


def get_joints_limits(body, joints, client=0):
    """
    Obtain the limits of a set of joints
    :param body: int
    :param joints: array type
    :return: a tuple of 2 arrays - lower limit and higher limit
    """
    lower_limit = []
    upper_limit = []
    for joint in joints:
        lower_limit.append(get_joint_info(body, joint, client).jointLowerLimit)
        upper_limit.append(get_joint_info(body, joint, client).jointUpperLimit)
    return lower_limit, upper_limit


def get_min_limit(body, joint, client=0):
    return get_joint_limits(body, joint, client)[0]


def get_max_limit(body, joint, client=0):
    return get_joint_limits(body, joint, client)[1]


def get_max_velocity(body, joint, client=0):
    return get_joint_info(body, joint, client).jointMaxVelocity


def get_max_force(body, joint, client=0):
    return get_joint_info(body, joint, client).jointMaxForce


def get_joint_q_index(body, joint, client=0):
    return get_joint_info(body, joint, client).qIndex


def get_joint_v_index(body, joint, client=0):
    return get_joint_info(body, joint, client).uIndex


def get_joint_axis(body, joint, client=0):
    return get_joint_info(body, joint, client).jointAxis


def get_joint_parent_frame(body, joint, client=0):
    joint_info = get_joint_info(body, joint, client)
    return joint_info.parentFramePos, joint_info.parentFrameOrn


def violates_limit(body, joint, value, client=0):
    if not is_circular(body, joint, client):
        lower, upper = get_joint_limits(body, joint, client)
        if (value < lower) or (upper < value):
            return True
    return False


def violates_limits(body, joints, values, client=0):
    return any(violates_limit(body, joint, value, client) for joint, value in zip(joints, values))


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)


def wrap_joint(body, joint, value):
    if is_circular(body, joint):
        return wrap_angle(value)
    return value


def get_difference_fn(body, joints, client=0):
    def fn(q2, q1):
        difference = []
        for joint, value2, value1 in zip(joints, q2, q1):
            difference.append(circular_difference(value2, value1)
                              if is_circular(body, joint, client) else (value2 - value1))
        return list(difference)

    return fn


def get_refine_fn(body, joints, num_steps=0, client=0):
    difference_fn = get_difference_fn(body, joints, client)
    num_steps = num_steps + 1

    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            q = tuple((1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q)
            yield q
            # TODO: should wrap these joints

    return fn


# Body and base

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])


# Bodies

def get_bodies(client=0):
    return [p.getBodyUniqueId(i, physicsClientId=client)
            for i in range(p.getNumBodies(physicsClientId=client))]


def get_body_info(body, client=0):
    return BodyInfo(*p.getBodyInfo(body, physicsClientId=client))


def get_base_name(body, client=0):
    return get_body_info(body, client).base_name.decode(encoding='UTF-8')


def get_body_name(body, client=0):
    return get_body_info(body, client).body_name.decode(encoding='UTF-8')


def get_name(body, client=0):
    name = get_body_name(body, client)
    if name == '':
        name = 'body'
    return '{}{}'.format(name, int(body))


def has_body(name, client=0):
    try:
        body_from_name(name, client)
    except ValueError:
        return False
    return True


def body_from_name(name, client=0):
    for body in get_bodies(client):
        if get_body_name(body, client) == name:
            return body
    raise ValueError(name)


def remove_body(body, client=0):
    return p.removeBody(body, physicsClientId=client)


def get_body_pos(body, client=0):
    return get_body_pose(body, client)[0]


def get_body_quat(body, client=0):
    return get_body_pose(body, client)[1]  # [x,y,z,w]


def set_pose(body, pose, client=0):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=client)


def set_point(body, point, client=0):
    set_pose(body, (point, get_body_quat(body)), client)


def set_quat(body, quat, client=0):
    set_pose(body, (get_body_pos(body), quat), client)


def is_rigid_body(body, client=0):
    for joint in get_joints(body, client):
        if is_movable(body, joint, client):
            return False
    return True


def is_fixed_base(body, client=0):
    return get_mass(body, client) == STATIC_MASS


def dump_body(body, client=0):
    print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
        body, get_body_name(body, client), is_rigid_body(body, client), is_fixed_base(body, client)))
    for joint in get_joints(body, client):
        print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Limits: {}'.format(
            joint, get_joint_name(body, joint, client), JOINT_TYPES[get_joint_type(body, joint, client)],
            is_circular(body, joint, client), get_joint_limits(body, joint, client)))
    print('Link id: {} | Name: {} | Mass: {}'.format(-1, get_base_name(body), get_mass(body)))
    for link in get_links(body, client):
        print('Link id: {} | Name: {} | Parent: {} | Mass: {}'.format(
            link, get_link_name(body, link, client), get_link_name(body, get_link_parent(body, link, client)),
            get_mass(body, link, client)))
        # print(get_joint_parent_frame(body, link))
        # print(map(get_data_geometry, get_visual_data(body, link)))
        # print(map(get_data_geometry, get_collision_data(body, link)))


def dump_world(client=0):
    for body in get_bodies(client):
        dump_body(body, client)
        print()


def remove_all_bodies(client=0):
    for i in get_body_ids(client):
        p.removeBody(i, physicsClientId=client)


def reset_body_base(body, pose, client=0):
    p.resetBasePositionAndOrientation(body, pose[0], pose[1], physicsClientId=client)


def get_body_infos(client=0):
    """ Return all body info in a list """
    return [get_body_info(i, client) for i in get_body_ids(client)]


def get_body_names(client=0):
    """ Return all body names in a list """
    return [bi.body_name for bi in get_body_infos(client)]


def get_body_id(name, client=0):
    return get_body_names(client).index(name)


def get_body_ids(client=0):
    return sorted([p.getBodyUniqueId(i, physicsClientId=client) for i in range(p.getNumBodies(physicsClientId=client))])


def get_body_pose(body, client=0):
    raw = p.getBasePositionAndOrientation(body, physicsClientId=client)
    position = list(raw[0])
    orn = list(raw[1])
    return [position, orn]


# Control

def control_joint(body, joint, value, client=0):
    return p.setJointMotorControl2(bodyUniqueId=body,
                                   jointIndex=joint,
                                   controlMode=p.POSITION_CONTROL,
                                   targetPosition=value,
                                   targetVelocity=0,
                                   maxVelocity=get_max_velocity(body, joint),
                                   force=get_max_force(body, joint),
                                   physicsClientId=client)


def control_joints(body, joints, positions, client=0):
    return p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL,
                                       targetPositions=positions,
                                       targetVelocities=[0.0] * len(joints),
                                       forces=[get_max_force(body, joint) for joint in joints],
                                       physicsClientId=client)


def forward_kinematics(body, joints, positions, eef_link=None, client=0):
    eef_link = get_num_joints(body, client) - 1 if eef_link is None else eef_link
    old_positions = get_joint_positions(body, joints, client)
    set_joint_positions(body, joints, positions, client)
    eef_pose = get_link_pose(body, eef_link, client)
    set_joint_positions(body, joints, old_positions, client)
    return eef_pose


def inverse_kinematics(body, eef_link, position, orientation=None, client=0):
    if orientation is None:
        jv = p.calculateInverseKinematics(bodyUniqueId=body,
                                          endEffectorLinkIndex=eef_link,
                                          targetPosition=position,
                                          residualThreshold=1e-3,
                                          physicsClientId=client)
    else:
        jv = p.calculateInverseKinematics(bodyUniqueId=body,
                                          endEffectorLinkIndex=eef_link,
                                          targetPosition=position,
                                          targetOrientation=orientation,
                                          residualThreshold=1e-3,
                                          physicsClientId=client)
    return jv


# Links

BASE_LINK = -1
STATIC_MASS = 0

get_num_links = get_num_joints
get_links = get_joints


def get_link_name(body, link, client=0):
    if link == BASE_LINK:
        return get_base_name(body, client)
    return get_joint_info(body, link, client).linkName.decode('UTF-8')


def get_link_parent(body, link, client=0):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link, client).parentIndex


def link_from_name(body, name, client=0):
    if name == get_base_name(body, client):
        return BASE_LINK
    for link in get_joints(body, client):
        if get_link_name(body, link, client) == name:
            return link
    raise ValueError(body, name)


def has_link(body, name, client=0):
    try:
        link_from_name(body, name, client)
    except ValueError:
        return False
    return True


LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])


def get_link_state(body, link, client=0):
    return LinkState(*p.getLinkState(body, link, physicsClientId=client))


def get_com_pose(body, link, client=0):  # COM = center of mass
    link_state = get_link_state(body, link, client)
    return list(link_state.linkWorldPosition), list(link_state.linkWorldOrientation)


def get_link_inertial_pose(body, link, client=0):
    link_state = get_link_state(body, link, client)
    return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation


def get_link_pose(body, link, client=0):
    if link == BASE_LINK:
        return get_body_pose(body, client)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link, client)
    return [list(link_state.worldLinkFramePosition), list(link_state.worldLinkFrameOrientation)]


def get_all_link_parents(body, client=0):
    return {link: get_link_parent(body, link, client) for link in get_links(body, client)}


def get_all_link_children(body, client=0):
    children = {}
    for child, parent in get_all_link_parents(body, client).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link, client=0):
    children = get_all_link_children(body, client)
    return children.get(link, [])


def get_link_ancestors(body, link, client=0):
    parent = get_link_parent(body, link, client)
    if parent is None:
        return []
    return get_link_ancestors(body, parent, client) + [parent]


def get_joint_ancestors(body, link, client=0):
    return get_link_ancestors(body, link, client) + [link]


def get_movable_joint_ancestors(body, link, client=0):
    return list(filter(lambda j: is_movable(body, j, client), get_joint_ancestors(body, link, client)))


def get_link_descendants(body, link, client=0):
    descendants = []
    for child in get_link_children(body, link, client):
        descendants.append(child)
        descendants += get_link_descendants(body, child, client)
    return descendants


def are_links_adjacent(body, link1, link2, client=0):
    return (get_link_parent(body, link1, client) == link2) or \
           (get_link_parent(body, link2, client) == link1)


def get_adjacent_links(body, client=0):
    adjacent = set()
    for link in get_links(body, client):
        parent = get_link_parent(body, link, client)
        adjacent.add((link, parent))
        # adjacent.add((parent, link))
    return adjacent


def get_adjacent_fixed_links(body, client=0):
    return list(filter(lambda item: not is_movable(body, item[0], client),
                       get_adjacent_links(body, client)))


def get_fixed_links(body, client=0):
    edges = defaultdict(list)
    for link, parent in get_adjacent_fixed_links(body, client):
        edges[link].append(parent)
        edges[parent].append(link)
    visited = set()
    fixed = set()
    for initial_link in get_links(body, client):
        if initial_link in visited:
            continue
        cluster = [initial_link]
        queue = deque([initial_link])
        visited.add(initial_link)
        while queue:
            for next_link in edges[queue.popleft()]:
                if next_link not in visited:
                    cluster.append(next_link)
                    queue.append(next_link)
                    visited.add(next_link)
        fixed.update(product(cluster, cluster))
    return fixed


DynamicsInfo = namedtuple('DynamicsInfo', ['mass', 'lateral_friction',
                                           'local_inertia_diagonal', 'local_inertial_pos', 'local_inertial_orn',
                                           'restitution', 'rolling_friction', 'spinning_friction',
                                           'contact_damping', 'contact_stiffness'])


def get_dynamics_info(body, link=BASE_LINK, client=0):
    return DynamicsInfo(*p.getDynamicsInfo(body, link, physicsClientId=client))


def get_mass(body, link=BASE_LINK, client=0):
    return get_dynamics_info(body, link, client).mass


def get_joint_inertial_pose(body, joint, client=0):
    dynamics_info = get_dynamics_info(body, joint, client)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn


# Contact and Collision


ContactPoint = namedtuple('ContactPoint', ['contactFlag', 'bodyUniqueIdA', 'bodyUniqueIdB', 'linkIndexA',
                                           'linkIndexB', 'positionOnA', 'positionOnB', 'contactNormalOnB',
                                           'contactDistance', 'normalForce', 'lateralFriction1', 'lateralFrictionDir1',
                                           'lateralFriction2', 'lateralFrictionDir2'])


def get_contact_potins(body_a, body_b, link_a, link_b, client=0):
    contact_points = p.getContactPoints(body_a, body_b, link_a, link_b, physicsClientId=client)
    return [ContactPoint(*point) for point in contact_points]


def get_closest_potins(body_a, body_b, distance, link_a, link_b, client=0):
    contact_points = p.getClosestPoints(body_a, body_b, distance, link_a, link_b, physicsClientId=client)
    return [ContactPoint(*point) for point in contact_points]



# Camera

CameraInfo = namedtuple('CameraInfo', ['width', 'height',
                                       'viewMatrix', 'projectionMatrix', 'cameraUp',
                                       'cameraForward', 'horizontal', 'vertical',
                                       'yaw', 'pitch', 'dist', 'target'])


def reset_camera(yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0), client=0):
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target, physicsClientId=client)


def get_camera(client=0):
    return CameraInfo(*p.getDebugVisualizerCamera(physicsClientId=client))


# Visualization

def create_frame_marker(pose=((0, 0, 0), (0, 0, 0, 1)),
                        x_color=np.array([1, 0, 0]),
                        y_color=np.array([0, 1, 0]),
                        z_color=np.array([0, 0, 1]),
                        line_length=0.1,
                        line_width=2,
                        life_time=0,
                        replace_frame_id=None,
                        client=0):
    """
    Create a pose marker that identifies a position and orientation in space with 3 colored lines.
    """
    position = np.array(pose[0])
    orientation = np.array(pose[1])

    pts = np.array([[0, 0, 0], [line_length, 0, 0], [0, line_length, 0], [0, 0, line_length]])
    rotIdentity = np.array([0, 0, 0, 1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0, :], rotIdentity, physicsClientId=client)
    px, _ = p.multiplyTransforms(position, orientation, pts[1, :], rotIdentity, physicsClientId=client)
    py, _ = p.multiplyTransforms(position, orientation, pts[2, :], rotIdentity, physicsClientId=client)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3, :], rotIdentity, physicsClientId=client)

    if replace_frame_id is not None:
        x_id = p.addUserDebugLine(po, px, x_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[0], physicsClientId=client)
        y_id = p.addUserDebugLine(po, py, y_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[1], physicsClientId=client)
        z_id = p.addUserDebugLine(po, pz, z_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[2], physicsClientId=client)
    else:
        x_id = p.addUserDebugLine(po, px, x_color, line_width, life_time, physicsClientId=client)
        y_id = p.addUserDebugLine(po, py, y_color, line_width, life_time, physicsClientId=client)
        z_id = p.addUserDebugLine(po, pz, z_color, line_width, life_time, physicsClientId=client)
    frame_id = (x_id, y_id, z_id)
    return frame_id


def create_arrow_marker(pose=((0, 0, 0), (0, 0, 0, 1)),
                        line_length=0.1,
                        arrow_length=0.01,
                        line_width=2,
                        arrow_width=6,
                        life_time=0,
                        color_index=0,
                        raw_color=None,
                        replace_frame_id=None,
                        client=0):
    """
    Create an arrow marker that identifies the z-axis of the end effector frame. Add a dot towards the positive direction.
    """

    position = np.array(pose[0])
    orientation = np.array(pose[1])
    color = raw_color if raw_color is not None else rgb_colors_1[color_index % len(rgb_colors_1)]

    pts = np.array([[0, 0, 0], [line_length, 0, 0], [0, line_length, 0], [0, 0, line_length]])
    z_extend = [0, 0, line_length + arrow_length]
    rotIdentity = np.array([0, 0, 0, 1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0, :], rotIdentity, physicsClientId=client)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3, :], rotIdentity, physicsClientId=client)
    pz_extend, _ = p.multiplyTransforms(position, orientation, z_extend, rotIdentity, physicsClientId=client)

    if replace_frame_id is not None:
        z_id = p.addUserDebugLine(po, pz, color, line_width, life_time,
                                  replaceItemUniqueId=replace_frame_id[2],
                                  physicsClientId=client)
        z_extend_id = p.addUserDebugLine(pz, pz_extend, color, arrow_width, life_time,
                                         replaceItemUniqueId=replace_frame_id[2],
                                         physicsClientId=client)
    else:
        z_id = p.addUserDebugLine(po, pz, color, line_width, life_time, physicsClientId=client)
        z_extend_id = p.addUserDebugLine(pz, pz_extend, color, arrow_width, life_time, physicsClientId=client)
    frame_id = (z_id, z_extend_id)
    return frame_id


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)

    def get_rgb(self, val):
        return self.cmap(self.norm(val))[:3]


def rgb(value, minimum=-1, maximum=1):
    """ for the color map https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map """
    assert minimum <= value <= maximum
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r / 255., g / 255., b / 255.


def plot_heatmap_bar(cmap_name, vmin=-1, vmax=-1):
    fig = plt.figure(figsize=(10, 2))
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = mpl.colorbar.ColorbarBase(plt.gca(), cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
    cb.set_label('Heatmap bar')
    plt.pause(0.001)


# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
rgb_colors_255 = [(230, 25, 75),  # red
                  (60, 180, 75),  # green
                  (255, 225, 25),  # yello
                  (0, 130, 200),  # blue
                  (245, 130, 48),  # orange
                  (145, 30, 180),  # purple
                  (70, 240, 240),  # cyan
                  (240, 50, 230),  # magenta
                  (210, 245, 60),  # lime
                  (250, 190, 190),  # pink
                  (0, 128, 128),  # teal
                  (230, 190, 255),  # lavender
                  (170, 110, 40),  # brown
                  (255, 250, 200),  # beige
                  (128, 0, 0),  # maroon
                  (170, 255, 195),  # lavender
                  (128, 128, 0),  # olive
                  (255, 215, 180),  # apricot
                  (0, 0, 128),  # navy
                  (128, 128, 128),  # grey
                  (0, 0, 0),  # white
                  (255, 255, 255)]  # black

rgb_colors_1 = np.array(rgb_colors_255) / 255.


def draw_line(start_pos, end_pos, rgb_color=(1, 0, 0), width=3, lifetime=0, client=0):
    lid = p.addUserDebugLine(lineFromXYZ=start_pos,
                             lineToXYZ=end_pos,
                             lineColorRGB=rgb_color,
                             lineWidth=width,
                             lifeTime=lifetime,
                             physicsClientId=client)
    return lid


def draw_circle_around_z_axis(centre, radius, rgb_color=(1, 0, 0), width=3, lifetime=0, num_divs=100, client=0):
    points = np.array(centre) + radius * np.array(
        [(np.cos(ang), np.sin(ang), 0) for ang in np.linspace(0, 2 * np.pi, num_divs)])
    lids = []
    for i in range(len(points) - 1):
        start_pos = points[i]
        end_pos = points[i + 1]
        lid = p.addUserDebugLine(lineFromXYZ=start_pos,
                                 lineToXYZ=end_pos,
                                 lineColorRGB=rgb_color,
                                 lineWidth=width,
                                 lifeTime=lifetime,
                                 physicsClientId=client)
        lids.append(lid)
    return lids


def draw_sphere_body(position, radius, rgba_color, client=0):
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=client)
    body_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id, physicsClientId=client)
    return body_id


def remove_marker(marker_id, client=0):
    p.removeUserDebugItem(marker_id, physicsClientId=client)


def remove_markers(marker_ids, client=0):
    for i in marker_ids:
        p.removeUserDebugItem(i, physicsClientId=client)


def remove_all_markers(client=0):
    p.removeAllUserDebugItems(physicsClientId=client)
