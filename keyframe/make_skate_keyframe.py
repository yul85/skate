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
    s00q[left_leg] = 0.1, 0., 0., -0.
    s00q[right_leg] = -0.1, -0., 0., -0.
    state00 = State("state00", 2.0, 0.0, 0.2, s00q)

    s_stable_q = np.zeros(skel.ndofs)
    s_stable_q[upper_body] = 0., 0., -0.4
    s_stable_q[spine] = 0.0, 0., 0.4
    s_stable_q[left_leg] = 0.1, 0., 0.3, -0.5
    s_stable_q[right_leg] = -0.1, -0., 0.3, -0.5
    s_stable_q[foot] = 0., 0., 0.2, -0., -0., 0.2
    state_stable = State("state_stable", 3.0, 2.2, 0.0, s_stable_q)

    s_force_q = np.zeros(skel.ndofs)
    s_force_q[upper_body] = 0., 0., -0.4
    s_force_q[spine] = 0.0, 0., 0.4
    s_force_q[left_leg] = 0.1, 0., 0.3, -0.5
    s_force_q[right_leg] = -0.1, -0., 0.3, -0.5
    s_force_q[foot] = 0., 0., 0.2, -0., -0., 0.2
    state_force = State("state_force", 5., 2.2, 0.0, s_force_q)

    s011q = np.zeros(skel.ndofs)
    s011q[upper_body] = 0., -0.3, -0.4
    s011q[spine] = 0.0, 0.0, 0.4
    s011q[left_leg] = 0.2, 0., 0.3, -0.3
    s011q[right_leg] = 0.1, -0., 0.3, -0.5
    # s1q[arms] = 0.5, -2.0
    # s1q[arms_y] = -0.7, 0.5
    s011q[foot] = 0., 0., 0.0, 0., -0., 0.2
    state011 = State("state011", 0.3, 2.2, 0.0, s011q)

    s1q = np.zeros(skel.ndofs)
    s1q[upper_body] = 0., -0.3, -0.4
    s1q[spine] = 0.0, 0.0, 0.4
    s1q[left_leg] = 0.1, 0., 0.9, -0.7
    s1q[right_leg] = 0., -0., 0.1, -0.2
    # s1q[arms] = 0.5, -2.0
    # s1q[arms_y] = -0.7, 0.5
    s1q[foot] = 0., 0., 0.2, 0., -0., 0.2
    state1 = State("state1", 0.3, 2.2, 0.0, s1q)

    s_lift_q = np.zeros(skel.ndofs)
    s_lift_q[upper_body] = 0., -0.3, -0.5
    # s_lift_q[spine] = 0.0, 0.0, 0.4
    s_lift_q[left_leg] = 0.1, 0., 0.5, -0.7
    s_lift_q[right_leg] = 0.3, -0., -0.3, -0.5
    s_lift_q[arms] = 0.5, -2.0
    s_lift_q[arms_y] = -0.7, 0.5
    s_lift_q[foot] = 0., 0., 0.2, -0., -0., 0.2
    state_lift = State("state_lift", 0.5, 2.2, 0.0, s_lift_q)

    s_low_q = np.zeros(skel.ndofs)
    s_low_q[upper_body] = 0., -0.3, -0.9
    # s_low_q[spine] = 0.0, 0.0, 0.4
    s_low_q[left_leg] = 0.1, 0., 0.2, -1.0
    s_low_q[right_leg] = 0.3, -0.0, -1.0, -0.7
    s_low_q[arms] = 0.5, -2.0
    s_low_q[arms_y] = -0.7, -0.5
    s_low_q[foot] = 0., 0., 0.8, -0., -0., -0.5
    state_low = State("state_low", 0.5, 2.2, 0.0, s_low_q)

    s01q = np.zeros(skel.ndofs)
    s01q[upper_body] = 0., -0.3, -0.9
    # s01q[spine] = 0.0, 0., 0.4
    s01q[left_leg] = -0.1, 0., 0.5, -1.5
    s01q[right_leg] = 0.3, -0., -1.2, -0.5
    # s01q[leg_y] = 0.4, -0.4
    s01q[arms] = 0.5, -2.0
    s01q[arms_y] = -1.0, -0.2
    # s01q[knee] = 0.05, -0.
    # s01q[foot] = -0., 0.785, 0.2, 0., -0.785, 0.2
    s01q[foot] = 0., 0., 1.0, -0., -0., -0.5
    state01 = State("state01", 0.2, 2.2, 0.0, s01q)

    s_air_q = np.zeros(skel.ndofs)
    s_air_q[upper_body] = 0., 0.3, 0.5
    s_air_q[left_leg] = -0.2, 0.3, 0.5, -0.5
    s_air_q[right_leg] = 0.2, 0.3, -0.1, 0.
    # s4[arms] = -0.5, -0.5
    s_air_q[arms_z] = 0.7, 0.7
    s_air_q[elbows] = -2.0, 2.0
    s_air_q[foot] = 0.0, 0.0, 0., 0.0, 0.0, -0.5
    state_air = State("state_air", 1.0, 2.0, 0.0, s_air_q)

    s_terminal_q = np.zeros(skel.ndofs)

    state_t = State("state_t", 50., 2.0, 2.0, s_terminal_q)

    state00.set_next(state_stable)
    state_stable.set_next(state_force)
    state_force.set_next(state011)
    state011.set_next(state1)
    state1.set_next(state_lift)
    state_lift.set_next(state_low)
    state_low.set_next(state01)
    state01.set_next(state_air)
    state_air.set_next(state_t)

    states = [state00, state_stable, state_force, state011, state1, state_lift, state_low, state01, state_air, state_t]
    filename = 'skating_jump.skkey'

    with open(filename, 'wb') as f:
        pickle.dump(states, f)

