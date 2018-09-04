import numpy as np

class motionPlan(object):
    def __init__(self, skel, T, x):
        self.skel = skel
        self.ndofs = skel.num_dofs()
        self.mp = []
        self.T = T
        m_length = self.ndofs * 2 + 3 * 4
        # m = np.zeros(m_length)
        for ii in range(0, T):
            self.mp.append(x[ii * m_length:(ii+1)*m_length])

    def get_pose(self, i):
        return self.mp[i][0:self.ndofs]

    def get_tau(self, i):
        return self.mp[i][self.ndofs:self.ndofs*2]

    # def get_COM_position(self, i):
    #     return self.mp[i][self.ndofs:self.ndofs+3]
    #
    # def get_end_effector_l_position(self, i):
    #     return self.mp[i][self.ndofs+3:self.ndofs+3+3]

    # def get_end_effector_r_position(self, i):
    #     return self.mp[i][self.ndofs+3+3:self.ndofs+3+3*2]

    def get_contact_force(self, i):
        return self.mp[i][self.ndofs*2:]

    # def get_end_effector_angular_speed(self, i):
    #     return self.mp[i][self.ndofs+3+3*2 + 3 * 4:self.ndofs+3+3*2 + 3 * 4 + 3*2]
    #
    # def get_end_effector_rotation_angle(self, i):
    #     return self.mp[i][self.ndofs+3+3*2 + 3 * 4 + 3*2:]

    # def get_end_effector_rotation_info(self, i):
    #
    #     angular_speed_l = self.mp[i][self.ndofs + 3 + 3 * 2+ 3 * 4:self.ndofs + 3 + 3 * 2 + 3 * 4 + 3]
    #     angular_speed_r = self.mp[i][self.ndofs + 3 + 3 * 2 + 3 * 4 + 3:self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2]
    #
    #     rotation_angle_l = self.mp[i][self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2:self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2 + 2]
    #     rotation_angle_r = self.mp[i][self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2 + 2:self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2 + 2 * 2]
    #
    #
    #     return self.mp[i][0:self.ndofs]


