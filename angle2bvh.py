import numpy as np
from PyCommon.modules.Math import mmMath as mm
import math
import os
import bvh
from scipy.spatial.transform import Rotation

class MakeBvh(object):
    def __init__(self):
        self.skel = None
        self.joint_name = None

    def angle2bvh(self):
        file_dir = 'data/mocap/movingcam/'
        # fn = 'bc_small'
        # fn = 'backflip_a'
        fn = 'jump_result'

        f_name = file_dir + fn + '_angle.txt'

        # line_num = 0

        # read joint angles of each key pose frame from txt file
        angles = []

        with open(f_name) as f:
            for line in f:
                line = line.replace(' \n', '')
                values = line.split(" ")
                # print(values, type(values))
                angle = []
                for val in values:
                    angle.append(float(val))
                # print("angle: ", type(angle))
                angles.append(np.asarray(angle))
                # line_num = line_num + 1

        # make bvh file

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
        with open('/home/yuri/PycharmProjects/skate/SkateUtils/skeleton.bvh', 'r') as f:
            skeleton_info = f.read()
            mybvh = bvh.Bvh(skeleton_info)
            self.joint_name = mybvh.get_joints_names()

        # print(self.joint_name)
        output_f_name = fn + ".bvh"

        with open(output_f_name, 'w') as f:
            f.write(skeleton_info)
            f.write('MOTION\r\n')
            f.write('Frames: '+str(len(angles))+'\r\n')
            f.write('Frame Time: 0.0333333\r\n')
            t_pose_angles = [0. for _ in range(63)]
            t_pose_angles[1] = 104.721
            f.write(' '.join(map(str, t_pose_angles))+'\r\n')

            smpl_names = [
                'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
                'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
                'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
                'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
                'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
                'Left_Finger', 'Right_Finger'
            ]

            index_list = [0, 2, 5, 8, 11, 3, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21, 1, 4, 7, 10]
            # index_list = [2, 5, 8, 11, 3, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21, 1, 4, 7, 10, 0]
            # 23 joints + root orientation-> root translation + orientation +19 joints
            # 69 + 3 dofs -> 57 dofs + 3+3

            for q in angles:
                euler_middle_q = np.asarray(q)
                # for joit_i in range(int(len(q) / 3)):
                #     if joit_i != 1:
                #         temp_axis_angle = np.asarray([q[3*joit_i], q[3*joit_i+1], q[3*joit_i+2]])
                #         r_offset = np.eye(3)
                #         if 'scapular_left' in skel.dof(3*joit_i).name:
                #             r_offset = mm.rotX(-0.9423)
                #         elif 'scapular_right' in skel.dof(3*joit_i).name:
                #             r_offset = mm.rotX(0.9423)
                #         elif 'bicep_left' in skel.dof(3*joit_i).name:
                #             r_offset = mm.rotX(-1.2423)
                #         elif 'bicep_right' in skel.dof(3*joit_i).name:
                #             r_offset = mm.rotX(1.2423)
                #         euler_result = axis2Euler(temp_axis_angle, r_offset)
                #         euler_middle_q[3*joit_i:3*joit_i+3] = euler_result

                # print("middle_q:", euler_middle_q)
                # convert axis angle to euler angle
                euler_q = np.zeros(60)
                for i in range(len(index_list)):
                    ci = 3*index_list[i]        # corresponding_index
                    # print(i, ci)
                    axis_angle = np.asarray([euler_middle_q[ci], euler_middle_q[ci+1], euler_middle_q[ci+2]])
                    # axis_angle = np.asarray([euler_middle_q[i], euler_middle_q[i + 1], euler_middle_q[i + 2]])
                    euler_q[3*i:3*i+3] = 180./math.pi*Rotation.from_rotvec(axis_angle).as_euler('ZXY')

                # for i in range(len(euler_middle_q)/3):
                #     temp = np.asarray([euler_middle_q[i], euler_middle_q[i+1], euler_middle_q[i+2]])
                #     euler_q[i:i+3] = Rotation.from_rotvec(temp).as_euler('ZXY')
                #     i = i + 3

                # euler_q[0:3] = euler_middle_q[0:3]
                # euler_q = np.zeros(len(q)+6)       # add two toes dofs (6 = 3x2)
                # euler_q[0:3] = np.dot(mm.rotY(-math.pi / 2.), q[3:6] * 100.) + np.array([0., 104.721, 0.])
                # euler_q[3:6] = euler_middle_q[0:3]
                # euler_q[6:15] = euler_middle_q[15:24]
                # euler_q[15:18] = np.zeros(3)
                # euler_q[18:51] = euler_middle_q[24:]
                # euler_q[51:60] = euler_middle_q[6:15]
                # euler_q[60:63] = np.zeros(3)

                t_euler_q = np.zeros(63)
                t_euler_q[1] = 104.721
                t_euler_q[3:63] = euler_q
                # t_euler_q[3:4] = -1.*t_euler_q[3:4]
                f.write(' '.join(map(str, t_euler_q)))
                f.write('\r\n')

if __name__ == '__main__':
    mybvh = MakeBvh()
    mybvh.angle2bvh()