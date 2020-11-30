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
    s00q[arms] = 0.785, -0.785
    # s00q[arms_z] = 0.785, 0.785
    state00 = State("state00", 2.0, 0.0, 0.2, s00q)

    s_stable_q = np.zeros(skel.ndofs)
    s_stable_q[upper_body] = 0., 0., -0.4
    # s_stable_q[spine] = 0.0, 0., 0.4
    s_stable_q[left_leg] = 0., 0., 0.3, -0.5
    s_stable_q[right_leg] = -0., -0., 0.3, -0.5
    s_stable_q[arms] = 0.785, -0.785
    s_stable_q[foot] = 0., 0., 0.2, -0., -0., 0.2
    state_stable = State("state_stable", 2.0, 2.2, 0.0, s_stable_q)

    s_force_q = np.zeros(skel.ndofs)
    s_force_q[upper_body] = 0., 0., -0.4
    # s_force_q[spine] = 0.0, 0., 0.4
    s_force_q[left_leg] = 0., 0., 0.3, -0.5
    s_force_q[right_leg] = -0., -0., 0.3, -0.5
    s_force_q[arms] = 0.785, -0.785
    s_force_q[foot] = 0., 0., 0.2, -0., -0., 0.2
    state_force = State("state_force", 1.0, 2.2, 0.0, s_force_q)

    s_turn_q = np.zeros(skel.ndofs)
    s_turn_q[upper_body] = 0., 0.2, -0.4
    # s_turn_q[spine] = 0.0, 0., 0.4
    s_turn_q[left_leg] = 0., 0., 0.3, -0.5
    s_turn_q[right_leg] = -0., -0., 0.3, -0.5
    s_turn_q[arms] = 0.785, -0.785
    s_turn_q[foot] = 0., 0., 0.2, -0., -0., 0.2
    state_turn = State("state_turn", 2.0, 2.2, 0.0, s_turn_q)

    # s1q = np.zeros(skel.ndofs)
    # s1q[upper_body] = 0.0, 0.0, -0.4
    # # s1q[spine] = 0.0, 0., 0.4
    # s1q[left_leg] = -0.2, 0., 0.3, -0.5
    # s1q[right_leg] = 0., -0., 0.5, -0.9
    # s1q[arms] = 0.785, -0.785
    # s1q[foot] = 0., 0., 0.2, -0., -0., 0.2
    # state1 = State("state1", 0.2, 0.0, 0.0, s1q)
    #
    # s2q = np.zeros(skel.ndofs)
    # s2q[upper_body] = 0.0, 0.0, -0.4
    # # s2q[spine] = 0.0, 0., 0.4
    # s2q[left_leg] = -0.2, 0., 0.3, -0.5
    # s2q[right_leg] = 0.2, -1.57, 0.3, -0.5
    # s2q[arms] = 0.785, -0.785
    # s2q[foot] = 0., 0., 0.2, -0., -0., 0.2
    # state2 = State("state2", 0.5, 0.0, 0.0, s2q)
    #
    # s3q = np.zeros(skel.ndofs)
    # s3q[upper_body] = 0.0, 0.0, -0.4
    # # s3q[spine] = 0.0, 0., 0.4
    # s3q[left_leg] = -0.2, 0., 0.3, -0.5
    # s3q[right_leg] = 0., -0.785, -0.3, -0.2
    # s3q[arms] = 0.785, -0.785
    # s3q[foot] = 0., 0., 0.2, -0., -0., -0.2
    # state3 = State("state3", 0.5, 0.0, 0.0, s3q)
    #
    # s_left_q = np.zeros(skel.ndofs)
    # s_left_q[upper_body] = 0., 0.5, -0.4
    # # s_left_q[spine] = 0.0, 0., 0.4
    # s_left_q[left_leg] = -0.2, 0., 0.3, -0.5
    # s_left_q[right_leg] = 0., -0.785, -0.3, -0.2
    # s_left_q[arms] = 0.785, -0.785
    # # s_left_q[arms] = 0.4, -0.4
    # # s_left_q[arms_y] = 1.0, 1.0
    # s_left_q[foot] = 0., 0., 0.2, -0., -0., 0.2
    # state_l = State("state_l", 1.0, 2.2, 0.0, s_left_q)
    #
    # s_right_q = np.zeros(skel.ndofs)
    # # s_right_q[upper_body] = 0., 0., -0.2
    # # s_right_q[spine] = 0.0, 0., 0.4
    # s_right_q[left_leg] = 0., 0., 0.0, -0.1
    # s_right_q[right_leg] = -0., -0., 0.0, -0.1
    # s_right_q[arms] = 0.4, -0.4
    # s_right_q[arms_y] = -1.0, -1.0
    # # s_right_q[leg_y] = 0.4, -0.4
    # s_right_q[foot] = 0., 0., 0.1, -0., -0., 0.1
    # state_r = State("state_r", 0.5, 2.2, 0.0, s_right_q)

    s_terminal_q = np.zeros(skel.ndofs)
    s_terminal_q[upper_body] = 0.0, -0., -0.1
    s_terminal_q[left_leg] = 0., 0., 0., -0.
    s_terminal_q[right_leg] = -0., -0., 0., -0.
    s_terminal_q[arms] = 0.785, -0.785
    # s_terminal_q[arms_z] = 0.785, 0.785
    state_t = State("state_t", 50., 2.0, 2.0, s_terminal_q)

    state00.set_next(state_stable)
    state_stable.set_next(state_force)
    state_force.set_next(state_turn)
    state_turn.set_next(state_t)

    states = [state00, state_stable, state_force, state_turn, state_t]
    filename = 'skating_turn_simple.skkey'

    with open(filename, 'wb') as f:
        pickle.dump(states, f)

