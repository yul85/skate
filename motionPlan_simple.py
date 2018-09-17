class motionPlan(object):
    def __init__(self, skel, T, x):
        self.skel = skel
        self.mp = []
        self.T = T
        self.m_length = 3 * 3 + 3 * 4 + 3
        # m = np.zeros(m_length)
        for ii in range(0, T):
            self.mp.append(x[ii * self.m_length:(ii+1)*self.m_length])

    def get_com_position(self, i):
        return self.mp[i][0:3]

    def get_end_effector_position(self, i):
        return self.mp[i][3:3*3]

    def get_end_effector_l_position(self, i):
        return self.mp[i][3:3*2]

    def get_end_effector_r_position(self, i):
        return self.mp[i][3*2:3*3]

    def get_contact_force(self, i):
        return self.mp[i][3*3:3*7]

    def get_angular_momentum(self, i):
        return self.mp[i][3*7:]

