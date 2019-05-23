import copy
import IKsolve_double_stance
import numpy as np
from scipy.optimize import minimize
from enum import IntEnum


class State(object):
    def __init__(self, name, dt, c_d, c_v, angles):
        self.name = name
        self.dt = dt
        self.c_d = c_d
        self.c_v = c_v
        self.angles = angles
        self.next = None

        self.ext_force = np.zeros(3)

    def set_next(self, next):
        self.next = next

    def get_next(self):
        return self.next if self.next is not None else self


class IKType(IntEnum):
    DOUBLE = 0
    LEFT = 1
    RIGHT = 2


class IKSolver(object):
    def __init__(self, skel, ik_type):
        self.ik_type = ik_type

        self.skel = skel
        self.h = 0.
        self.target_foot = self.skel.C + self.skel.dC * self.h
        self.prev_state = -1

        self.offset_list = [[-0.1040 + 0.0216, -0.027-0.0216, 0.0], [0.1040+0.0216, -0.027-0.0216, 0.0]]

    def set_ik_type(self, ik_type):
        self.ik_type = ik_type

    def update_target(self, ground_height):
        self.target_foot_left1 = self.skel.body("h_blade_left").to_world(self.offset_list[0])
        self.target_foot_left2 = self.skel.body("h_blade_left").to_world(self.offset_list[1])
        self.target_foot_right1 = self.skel.body("h_blade_right").to_world(self.offset_list[0])
        self.target_foot_right2 = self.skel.body("h_blade_right").to_world(self.offset_list[1])

        self.target_foot_left1[1] = ground_height
        self.target_foot_left2[1] = ground_height
        self.target_foot_right1[1] = ground_height
        self.target_foot_right2[1] = ground_height

        self.target_foot_vel = self.skel.dC
        self.target_foot_vel[1] = 0

    def set_params(self, x):
        q = self.skel.q
        q[6:] = x
        self.skel.set_positions(q)

    def f(self, x):
        self.set_params(x)

        foot_state_left1 = self.skel.body("h_blade_left").to_world(self.offset_list[0])
        foot_state_left2 = self.skel.body("h_blade_left").to_world(self.offset_list[1])
        foot_state_right1 = self.skel.body("h_blade_right").to_world(self.offset_list[0])
        foot_state_right2 = self.skel.body("h_blade_right").to_world(self.offset_list[1])
        left_foot_f = 0.5 * (np.linalg.norm(foot_state_left1 - self.target_foot_left1) ** 2 + np.linalg.norm(foot_state_left2 - self.target_foot_left2) ** 2)
        right_foot_f = 0.5 * (np.linalg.norm(foot_state_right1 - self.target_foot_right1) ** 2 + np.linalg.norm(foot_state_right2 - self.target_foot_right2) ** 2)

        return_f = 0.
        if self.ik_type in [IKType.DOUBLE, IKType.LEFT]:
            return_f += left_foot_f

        if self.ik_type in [IKType.DOUBLE, IKType.RIGHT]:
            return_f += right_foot_f

        return return_f

        # return 0.5 * (np.linalg.norm(foot_state_right1 - self.target_foot_right1) ** 2 + np.linalg.norm(foot_state_right2 - self.target_foot_right2) ** 2)
        # return 0.5 * (np.linalg.norm(foot_state_left1 - self.target_foot_left1) ** 2 + np.linalg.norm(foot_state_left2 - self.target_foot_left2) ** 2)

    # def g(self, x):
    #     self.set_params(x)
    #
    #     foot_state_left1 = self.skel.body("h_blade_left").to_world(self.offset_list[0])
    #     foot_state_left2 = self.skel.body("h_blade_left").to_world(self.offset_list[1])
    #     foot_state_right1 = self.skel.body("h_blade_right").to_world(self.offset_list[0])
    #     foot_state_right2 = self.skel.body("h_blade_right").to_world(self.offset_list[1])
    #
    #     J_foot_left1 = self.skel.body("h_blade_left").linear_jacobian(self.offset_list[0])
    #     J_foot_left2 = self.skel.body("h_blade_left").linear_jacobian(self.offset_list[1])
    #     J_foot_right1 = self.skel.body("h_blade_right").linear_jacobian(self.offset_list[0])
    #     J_foot_right2 = self.skel.body("h_blade_right").linear_jacobian(self.offset_list[1])
    #     AA1 = np.append(foot_state_left1 - self.target_foot_left1, foot_state_left2 - self.target_foot_left2)
    #     AA2 = np.append(foot_state_right1 - self.target_foot_right1, foot_state_right2 - self.target_foot_right2)
    #     AA = np.append(AA1, AA2)
    #
    #     J1 = np.vstack((J_foot_left1, J_foot_left2))
    #     J2 = np.vstack((J_foot_right1, J_foot_right2))
    #     J = np.vstack((J1, J2))
    #
    #     g = AA.dot(J)
    #
    #     # g = AA2.dot(J2)     # only right foot
    #     # g = AA1.dot(J1)  # only right foot
    #
    #     return g[6:]

    def solve(self, ):
        q_backup = self.skel.q
        res = minimize(self.f, x0=self.skel.q[6:], method="SLSQP")
        # res = minimize(self.f, x0=self.skel.q[6:], jac=self.g, method="SLSQP")

        self.skel.set_positions(q_backup)

        return res['x']


def _revise_pose(ik, ref_skel, pose, ground_height):
    ik_res = np.asarray(ref_skel.q)
    if pose is not None:
        ref_skel.set_positions(pose.angles)
        ik_res = copy.deepcopy(pose.angles)

    ik.update_target(ground_height)
    ik_res[6:] = ik.solve()
    if pose is not None:
        pose.angles = ik_res
    else:
        ref_skel.set_positions(ik_res)


def revise_pose(ref_skel, pose, ik_type=IKType.DOUBLE, ground_height=None):
    """

    :param ref_skel:
    :type ref_skel: pydart.Skeleton
    :param pose:
    :type pose: State | None
    :param ik_type:
    :type ik_type: IKType
    :param ground_height:
    :return:
    """
    ik = IKSolver(ref_skel, ik_type)

    if ground_height is None:
        if ik_type == IKType.DOUBLE:
            ground_height = max(ref_skel.body('h_blade_right').to_world((-0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                                ref_skel.body('h_blade_right').to_world((+0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                                ref_skel.body('h_blade_left').to_world((-0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                                ref_skel.body('h_blade_left').to_world((+0.104 + 0.0216, -0.027 - 0.0216, 0.))[1]
                                )
        elif ik_type == IKType.LEFT:
            ground_height = max(
                ref_skel.body('h_blade_left').to_world((-0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                ref_skel.body('h_blade_left').to_world((+0.104 + 0.0216, -0.027 - 0.0216, 0.))[1]
            )
        elif ik_type == IKType.RIGHT:
            ground_height = max(ref_skel.body('h_blade_right').to_world((-0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                                ref_skel.body('h_blade_right').to_world((+0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                                )

    _revise_pose(ik, ref_skel, pose, ground_height)
