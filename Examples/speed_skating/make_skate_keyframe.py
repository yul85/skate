from SkateUtils.KeyPoseState import State
import numpy as np


def make_keyframe(skel):
    """

    :param skel:
    :type skel: pydart.Skeleton
    :return:
    """

    pelvis_pos_y = skel.dof_indices(["j_pelvis_pos_y"])
    pelvis_x = skel.dof_indices(["j_pelvis_rot_x"])
    pelvis = skel.dof_indices(["j_pelvis_rot_y", "j_pelvis_rot_z"])
    upper_body = skel.dof_indices(["j_abdomen_x", "j_abdomen_y", "j_abdomen_z"])
    spine = skel.dof_indices(["j_spine_x", "j_spine_y", "j_spine_z"])
    right_leg = skel.dof_indices(["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right_z"])
    left_leg = skel.dof_indices(["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z", "j_shin_left_z"])
    knee = skel.dof_indices(["j_shin_left_x", "j_shin_right_x"])
    arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
    arms_y = skel.dof_indices(["j_bicep_left_y", "j_bicep_right_y"])
    arms_z = skel.dof_indices(["j_bicep_left_z", "j_bicep_right_z"])
    foot = skel.dof_indices(["j_heel_left_x", "j_heel_left_y", "j_heel_left_z", "j_heel_right_x", "j_heel_right_y", "j_heel_right_z"])
    leg_y = skel.dof_indices(["j_thigh_left_y", "j_thigh_right_y"])
    # blade = skel.dof_indices(["j_heel_right_2"])

    # ===========pushing side to side new===========

    s00q = np.zeros(skel.ndofs)

    s00q[upper_body] = 0.0, -0., -0.5
    s00q[left_leg] = 0.05, 0., 0.5, -0.7
    s00q[right_leg] = -0.5, -0., 0.5, -0.9
    s00q[leg_y] = 0., -0.785
    s00q[foot] = 0.03, 0., 0.2, -0., -0., 0.4
    state00 = State("State0", 2.0, 0.0, 0.2, s00q)

    s01q = np.zeros(skel.ndofs)
    s01q[upper_body] = 0., 0.2, -0.3
    s01q[left_leg] = 0.3, 0., 1.5, -1.7
    s01q[right_leg] = -0.3, -0., 0.5, -0.9
    s01q[leg_y] = 0.4, -0.4
    s01q[foot] = 0., 0.4, 0.5, -0.1, -0.4, 0.4
    state01 = State("State1", 0.5, 2.2, 0.0, s01q)

    s02q = np.zeros(skel.ndofs)
    s02q[upper_body] = 0., -0., -0.5
    s02q[left_leg] = 0.1, 0., 0.5, -1.
    s02q[right_leg] = -0.5, -0., 1.0, -2.0
    s02q[leg_y] = 0.4, -0.
    s02q[foot] = 0.1, 0.4, 0.5, -0., -0., 1.0
    state02 = State("State2", 0.5, 2.2, 0.0, s02q)

    s03q = np.zeros(skel.ndofs)
    s03q[upper_body] = 0., -0.2, -0.5
    s03q[left_leg] = 0.5, 0., 1.0, -2.0
    s03q[right_leg] = -0.1, -0., 0.3, -1.0
    s03q[leg_y] = 0., -0.4
    s03q[foot] = 0., 0., 1.0, -0.1, -0.4, 0.5
    state03 = State("State3", 0.5, 2.2, 0.0, s03q)

    state00.set_next(state01)
    state01.set_next(state02)
    state02.set_next(state03)
    state03.set_next(state02)

    return state00


if __name__ == '__main__':
    import pydart2 as pydart
    import pickle

    pydart.init()
    world = pydart.World(1./1200., '../../data/skel/skater_3dof_with_ground.skel')
    state = make_keyframe(world.skeletons[1])
    states = [state]
    for i in range(3):
        states.append(state.get_next())
        state = state.get_next()

    with open('speed_skating.skkey', 'wb') as f:
        pickle.dump(states, f)

