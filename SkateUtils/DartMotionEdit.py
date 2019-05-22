import os
import pydart2 as pydart
import numpy as np
import copy
import math
from scipy.spatial.transform import Rotation
from PyCommon.modules.Math import mmMath as mm


class DartSkelMotion(object):
    def __init__(self):
        self.qs = []
        self.dqs = []
        self.fps = 30.

    def __len__(self):
        assert len(self.qs) == len(self.dqs)
        return len(self.qs)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.fps = 1./float(f.readline().split(' ')[-1])
            for s in f.read().splitlines():
                ss = s.replace('[', '').split(']')
                sq, sdq = list(map(float, ss[0].split(','))), list(map(float, ss[1].split(',')))
                self.qs.append(np.asarray(sq))
                self.dqs.append(np.asarray(sdq))

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('Frame Times: '+str(1./self.fps))
            f.write('\n')
            for frame in range(len(self.qs)):
                f.write(str([d for d in np.asarray(self.qs[frame])]))
                f.write(str([d for d in np.asarray(self.dqs[frame])]))
                f.write('\n')

    def append(self, _q, _dq):
        self.qs.append(copy.deepcopy(_q))
        self.dqs.append(copy.deepcopy(_dq))

    def extend(self, frame, _qs, _dqs):
        qs = copy.deepcopy(_qs)
        dqs = copy.deepcopy(_dqs)
        offset_x = self.qs[frame][3] - qs[0][3]
        offset_z = self.qs[frame][5] - qs[0][5]
        del self.qs[frame:]
        del self.dqs[frame:]

        self.qs.extend(qs)
        self.dqs.extend(dqs)

        for i in range(len(qs)):
            self.qs[frame+i][3] += offset_x
            self.qs[frame+i][5] += offset_z

    def translate_by_offset(self, offset):
        for i in range(len(self.qs)):
            self.qs[i][3] += offset[0]
            self.qs[i][5] += offset[2]

    def get_q(self, frame):
        return self.qs[frame]

    def get_dq(self, frame):
        return self.dqs[frame]

    def refine_dqs(self, skel, start_frame=0):
        for frame in range(start_frame, len(self.dqs)):
            if frame == len(self.dqs)-1:
                self.dqs[frame] = np.asarray(skel.position_differences(self.qs[frame], self.qs[frame-1])) * self.fps
            else:
                self.dqs[frame] = np.asarray(skel.position_differences(self.qs[frame+1], self.qs[frame])) * self.fps


def axis2Euler(vec, offset_r=np.eye(3)):
    r = np.dot(offset_r, Rotation.from_rotvec(vec).as_dcm())
    r_after = np.dot(np.dot(mm.rotY(-math.pi/2.), r), mm.rotY(-math.pi/2.).T)
    return Rotation.from_dcm(r_after).as_euler('ZXY', True)


def skelqs2bvh(f_name, skel, qs):
    #make bvh file

    # 0:6         #pelvis
    # 15, 16, 17 # right leg
    # 18, 19, 20
    # 21, 22, 23
    # zero for toes
    # 24, 25, 26  #spine
    # 27, 28, 29
    # 30, 31, 32
    # 33, 34, 35, # left arm
    # 36, 37, 38
    # 39, 40, 41
    # 42, 43, 44
    # 45, 46, 47  #right arm
    # 48, 49, 50
    # 51, 52, 53
    # 54, 55, 56
    # 6, 7, 8     #left leg
    # 9, 10, 11
    # 12, 13, 14
    # zero for toes

    with open('/'.join(__file__.split('/')[:-1])+'/skeleton.bvh', 'r') as f:
        skeleton_info = f.read()

    with open(f_name, 'w') as f:
        f.write(skeleton_info)
        f.write('MOTION\r\n')
        f.write('Frames: '+str(len(qs))+'\r\n')
        f.write('Frame Time: 0.0333333\r\n')
        t_pose_angles = [0. for _ in range(len(qs[0])+6)]
        t_pose_angles[1] = 104.721
        f.write(' '.join(map(str, t_pose_angles))+'\r\n')

        for q in qs:
            euler_middle_q = np.asarray(skel.q)
            for joit_i in range(int(len(q) / 3)):
                if joit_i != 1:
                    temp_axis_angle = np.asarray([q[3*joit_i], q[3*joit_i+1], q[3*joit_i+2]])
                    r_offset = np.eye(3)
                    if 'scapular_left' in skel.dof(3*joit_i).name:
                        r_offset = mm.rotX(-0.9423)
                    elif 'scapular_right' in skel.dof(3*joit_i).name:
                        r_offset = mm.rotX(0.9423)
                    elif 'bicep_left' in skel.dof(3*joit_i).name:
                        r_offset = mm.rotX(-1.2423)
                    elif 'bicep_right' in skel.dof(3*joit_i).name:
                        r_offset = mm.rotX(1.2423)
                    euler_result = axis2Euler(temp_axis_angle, r_offset)
                    euler_middle_q[3*joit_i:3*joit_i+3] = euler_result

            # temp_axis_angle = np.asarray([env.skel.q[6], env.skel.q[7], env.skel.q[8]])
            # print("test vec:", temp_axis_angle)
            # euler_result = axis2Euler(temp_axis_angle)
            # print("euler angle: ", euler_result)

            # print("middle_q:", euler_middle_q)
            euler_q = np.zeros(len(q)+6)       # add two toes dofs (6 = 3x2)
            euler_q[0:3] = np.dot(mm.rotY(-math.pi / 2.), q[3:6] * 100.) + np.array([0., 104.721, 0.])
            euler_q[3:6] = euler_middle_q[0:3]
            euler_q[6:15] = euler_middle_q[15:24]
            euler_q[15:18] = np.zeros(3)
            euler_q[18:51] = euler_middle_q[24:]
            euler_q[51:60] = euler_middle_q[6:15]
            euler_q[60:63] = np.zeros(3)

            f.write(' '.join(map(str, euler_q)))
            f.write('\r\n')


if __name__ == '__main__':
    # mo = DartSkelMotion()
    # mo.load('skate.skmo')
    skelqs2bvh('test.bvh', None, [])
