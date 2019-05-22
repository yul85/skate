import os
from fltk import Fl
from skate_cma.skate_env2 import SkateDartEnv
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np
import pickle
import math

from scipy.spatial.transform import Rotation

import pydart2 as pydart

def axis2Euler(vec):
    return Rotation.from_rotvec(vec).as_euler('ZXY', True)

def main():
    MOTION_ONLY = False
    np.set_printoptions(precision=5)

    pydart.init()

    env_name = 'jump0507_2'
    max_time = 2.

    with open(env_name + '.skkey', 'rb') as skkey_file:
        skkey_states = pickle.load(skkey_file)

    angles = []
    count = 0
    state = skkey_states[0]
    state_duration = []
    while count < int(max_time * 10.):
        state_count = 0
        for _ in range(int(state.dt/0.1)):
            angles.append(state.angles[6:])
            state_count += 1
            count += 1
            if count == int(max_time * 10.):
                break
        state_duration.append(state_count)
        state = state.get_next()
    angles.append(angles[-1])

    env = SkateDartEnv(env_name)

    q = [skkey_states[0].angles.copy()]
    dq = [np.zeros_like(q[0])]
    # dq[0][3] = 1.

    x0t = np.zeros_like(q[0][6:])
    frame_offset = [0]
    cumulative_frame_duration = [sum(state_duration[:i]*3) for i in range(len(state_duration)+1)]

    x = [x0t]

    file_path = 'jump_inplace_solution/xbest.skcma'
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        state_list_in_file = list(map(int, [line.split()[0] for line in lines]))
        for i in range(len(state_duration)):
            if i in state_list_in_file:
                state_index_in_file = state_list_in_file.index(i)
                x_state = np.asarray(list(map(float, lines[state_index_in_file].split()[1:])))
                x.extend(np.split(x_state, state_duration[i]))
            else:
                x.extend([np.zeros_like(x0t) for _ in range(state_duration[i])])

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    rd_COM = [None]
    dart_world = env.world
    viewer_w, viewer_h = 1280, 720
    viewer = hsv.hpSimpleViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55), viewForceWnd=False)
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(env.ref_world, (194,207,245), yr.POLYGON_FILL))
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL))
        viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        viewer.doc.addRenderer('COM projection', yr.PointsRenderer(rd_COM))

    def simulateCallback(frame):
        if 0 in [ii - frame for ii in cumulative_frame_duration]:
            index = [ii - frame for ii in cumulative_frame_duration].index(0)
            duration = state_duration[index]
            env.update_ref_states(
                x[sum(state_duration[:index]):sum(state_duration[:index+1])+1],
                angles[sum(state_duration[:index]):sum(state_duration[:index+1])+1],
                q[0], dq[0], duration)
            frame_offset[0] = cumulative_frame_duration[index]

        env.step((frame-frame_offset[0])/3)
        q[0] = np.asarray(env.skel.q)
        dq[0] = np.asarray(env.skel.dq)

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

        euler_middle_q = np.asarray(env.skel.q)
        for joit_i in range(int(env.skel.num_dofs() / 3)):
            if joit_i != 1:
                temp_axis_angle = np.asarray([env.skel.q[3*joit_i], env.skel.q[3*joit_i+1], env.skel.q[3*joit_i+2]])
                euler_result = axis2Euler(temp_axis_angle)
                euler_middle_q[3*joit_i:3*joit_i+3] = euler_result

        # temp_axis_angle = np.asarray([env.skel.q[6], env.skel.q[7], env.skel.q[8]])
        # print("test vec:", temp_axis_angle)
        # euler_result = axis2Euler(temp_axis_angle)
        # print("euler angle: ", euler_result)

        # print("middle_q:", euler_middle_q)
        euler_q = np.zeros(env.skel.num_dofs()+6)       # add two toes dofs (6 = 3x2)
        euler_q[0:3] = env.skel.q[3:6] * 100.
        euler_q[3:6] = euler_middle_q[0:3]
        euler_q[6:15] = euler_middle_q[15:24]
        euler_q[15:18] = np.zeros(3)
        euler_q[18:51] = euler_middle_q[24:]
        euler_q[51:60] = euler_middle_q[6:15]
        euler_q[60:63] = np.zeros(3)

        # print("euler_q:", euler_q)

        f_name = 'jump_inplace.bvh'
        with open(f_name, 'a') as f:
            f.write(' '.join(map(str, euler_q)))
            f.write('\r\n')

        # print("f: ", frame)
        # contact rendering
        contacts = env.world.collision_result.contacts
        del rd_contact_forces[:]
        del rd_contact_positions[:]
        for contact in contacts:
            if contact.skel_id1 == 0:
                rd_contact_forces.append(-contact.f/1000.)
            else:
                rd_contact_forces.append(contact.f/1000.)
            rd_contact_positions.append(contact.p)

        del rd_COM[:]
        com = env.skel.com()
        com[1] = 0.
        rd_COM.append(com)
        # print(env.skel.com())

    viewer.setSimulateCallback(simulateCallback)
    viewer.setMaxFrame(cumulative_frame_duration[-1]-1)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()


if __name__ == '__main__':
    main()
