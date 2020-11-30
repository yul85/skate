from SkateUtils.KeyPoseState import State
import numpy as np
import pydart2 as pydart
import pickle


if __name__ == '__main__':
    pydart.init()
    world = pydart.World(1./1200., '../data/skel/skater_3dof_with_ground.skel')
    skel = world.skeletons[1]

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
    elbows = skel.dof_indices(["j_forearm_left_x", "j_forearm_right_x"])

    foot = skel.dof_indices(["j_heel_left_x", "j_heel_left_y", "j_heel_left_z", "j_heel_right_x", "j_heel_right_y", "j_heel_right_z"])
    leg_y = skel.dof_indices(["j_thigh_left_y", "j_thigh_right_y"])
    # blade = skel.dof_indices(["j_heel_right_2"])

    # ===========pushing side to side new===========

    s00q = np.zeros(skel.ndofs)
    s00q[upper_body] = 0.0, -0., -0.1
    s00q[left_leg] = 0., 0., 0., -0.
    s00q[right_leg] = -0., -0., 0., -0.
    state00 = State("state00", 2.0, 0.0, 0.2, s00q)

    s_stable_q = np.zeros(skel.ndofs)
    # s_stable_q[upper_body] = 0., 0., -0.4
    # s_stable_q[spine] = 0.0, 0., 0.4
    s_stable_q[left_leg] = 0., 0., 0.3, -0.5
    s_stable_q[right_leg] = -0., -0., 0.3, -0.5
    s_stable_q[foot] = 0., 0., 0.2, -0., -0., 0.2
    state_stable = State("state_stable", 3.0, 2.2, 0.0, s_stable_q)

    s_forward_q = np.zeros(skel.ndofs)
    s_forward_q[upper_body] = 0., 0., -0.2
    # s_forward_q[spine] = 0.0, 0., 0.4
    s_forward_q[left_leg] = 0., 0., 0.3, -0.5
    s_forward_q[right_leg] = -0., -0., 0.3, -0.5
    s_forward_q[leg_y] = 0.4, -0.4
    s_forward_q[foot] = 0., 0.4, 0.2, -0., -0.4, 0.2
    state_forward = State("state_forward", 0.5, 2.2, 0.0, s_forward_q)

    s_backward_q = np.zeros(skel.ndofs)
    # s_backward_q[upper_body] = 0., 0., 0.2
    s_backward_q[spine] = 0.0, 0., 0.4
    s_backward_q[left_leg] = 0., 0., 0.3, -0.5
    s_backward_q[right_leg] = -0., -0., 0.3, -0.5
    s_backward_q[leg_y] = -0.4, 0.4
    s_backward_q[foot] = 0., -0.4, 0.2, -0., 0.4, 0.2
    state_backward = State("state_backward", 0.5, 2.2, 0.0, s_backward_q)

    s_terminal_q = np.zeros(skel.ndofs)

    state_t = State("state_t", 50., 2.0, 2.0, s_terminal_q)

    state00.set_next(state_forward)
    state_forward.set_next(state_backward)
    state_backward.set_next(state_t)

    states = [state00, state_forward, state_backward, state_t]
    filename = 'skating_swizzle_simple.skkey'

    with open(filename, 'wb') as f:
        pickle.dump(states, f)

