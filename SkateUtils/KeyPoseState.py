import copy
import IKsolve_double_stance


class State(object):
    def __init__(self, name, dt, c_d, c_v, angles):
        self.name = name
        self.dt = dt
        self.c_d = c_d
        self.c_v = c_v
        self.angles = angles
        self.next = None

    def set_next(self, next):
        self.next = next

    def get_next(self):
        return self.next


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
