import pydart2 as pydart
import numpy as np
import copy
import IKsolve_double_stance


class State(object):
    def __init__(self, number, name, dt, c_d, c_v, angles):
        self.name = name
        self.dt = dt
        self.c_d = c_d
        self.c_v = c_v
        self.angles = angles
        self.next = None
        self.number = number

    def set_next(self, next):
        self.next = next

    def get_next(self):
        return self.next

    def get_number(self):
        return self.number


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
    state00 = State(0, "state00", 2.0, 0.0, 0.2, s00q)

    s01q = np.zeros(skel.ndofs)
    s01q[upper_body] = 0., 0.2, -0.3
    s01q[left_leg] = 0.3, 0., 1.5, -1.7
    s01q[right_leg] = -0.3, -0., 0.5, -0.9
    s01q[leg_y] = 0.4, -0.4
    s01q[foot] = 0., 0.4, 0.5, -0.1, -0.4, 0.4
    state01 = State(1, "state01", 0.5, 2.2, 0.0, s01q)

    s012q = np.zeros(skel.ndofs)
    s012q[upper_body] = 0., -0., -0.5
    s012q[left_leg] = 0.1, 0., 0.5, -1.
    s012q[right_leg] = -0.5, -0., 1.0, -2.0
    s012q[leg_y] = 0.4, -0.
    s012q[foot] = 0.1, 0.4, 0.5, -0., -0., 1.0
    state012 = State(2, "state012", 0.5, 2.2, 0.0, s012q)

    s02q = np.zeros(skel.ndofs)
    s02q[upper_body] = 0., -0.2, -0.5
    s02q[left_leg] = 0.5, 0., 1.0, -2.0
    s02q[right_leg] = -0.1, -0., 0.3, -1.0
    s02q[leg_y] = 0., -0.4
    s02q[foot] = 0., 0., 1.0, -0.1, -0.4, 0.5
    state02 = State(3, "state02", 0.5, 2.2, 0.0, s02q)

    s022q = np.zeros(skel.ndofs)
    s022q[upper_body] = 0., 0., -0.5
    s022q[left_leg] = 0.5, 0., 0.5, -0.7
    s022q[right_leg] = -0., -0., 1.0, -2.0
    s022q[leg_y] = 0.4, -0.
    s022q[foot] = 0.1, 0.4, 0.2, -0., -0., 1.0
    state022 = State(4, "state022", 0.5, 2.2, 0.0, s022q)

    s_terminal_q = np.zeros(skel.ndofs)

    state_t = State(5, "state_t", 50., 2.0, 2.0, s_terminal_q)

    state00.set_next(state01)
    state01.set_next(state012)
    state012.set_next(state02)
    state02.set_next(state012)

    # return [state00, state01, state012, state02, state012, state02, state012, state02, state_t]
    return state00


def revise_pose(ref_skel, pose):

    ik = IKsolve_double_stance.IKsolver(ref_skel, ref_skel, ref_skel.world.dt)
    ref_skel.set_positions(pose.angles)
    ground_height = max(ref_skel.body('h_blade_right').to_world((-0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                        ref_skel.body('h_blade_right').to_world((+0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                        ref_skel.body('h_blade_left').to_world((-0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                        ref_skel.body('h_blade_left').to_world((+0.104 + 0.0216, -0.027 - 0.0216, 0.))[1]
                        )

    ik_res = copy.deepcopy(pose.angles)
    ik.update_target(ground_height)
    ik_res[6:] = ik.solve()
    pose.angles = ik_res