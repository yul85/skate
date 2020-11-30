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
from PyCommon.modules.Math import mmMath as mm

from SkateUtils.DartMotionEdit import skelqs2bvh

def axis2Euler(vec):
    r = Rotation.from_rotvec(vec).as_dcm()
    r_after = np.dot(np.dot(mm.rotY(-math.pi/2.), r), mm.rotY(-math.pi/2.).T)
    return Rotation.from_dcm(r_after).as_euler('ZXY', True)
    # return Rotation.from_rotvec(vec).as_euler('ZXY', True)

def main():
    MOTION_ONLY = False
    np.set_printoptions(precision=5)

    pydart.init()

    # env_name = 'hmr_skating_spiral1'
    # env_name = 'sfv_spiral2_global'
    env_name = 'spiral4_handmade'
    max_time = 3.0

    with open(env_name + '.skkey', 'rb') as skkey_file:
        skkey_states = pickle.load(skkey_file)

    # for bvh file
    bvh_qs = []
    bvh_file_name = 'spiral1.bvh'

    angles = []
    count = 0
    state = skkey_states[0]
    state_duration = []
    while count < int(max_time * 10.):
        state_count = 0
        for _ in range(int(state.dt*10)):
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
    # dq[0][5] = 1.
    # dq[0][3] = 0.10976229*2.0
    # dq[0][5] = 0.17668021*2.0
    # dq[0][3] = 0.09747007 * 2.0
    # dq[0][5] = 0.18374869 * 2.0
    dq[0][3] = 0.20769821*5.
    dq[0][5] = 0.00536799*5.
    #0.09747007 0.         0.18374869
    x0t = np.zeros_like(q[0][6:])
    frame_offset = [0]
    cumulative_frame_duration = [sum(state_duration[:i]*3) for i in range(len(state_duration)+1)]

    x = [x0t]

    # file_path = 'spiral4_handmade_model_202003021927/xbest.skcma'
    file_path = 'spiral4_handmade_model_202003031616/xbest.skcma'

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

    p1_x = []
    p2_x = []

    dart_world = env.world
    viewer_w, viewer_h = 1280, 720
    viewer = hsv.hpSimpleViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55), viewForceWnd=False)
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(env.ref_world, (194,207,245), yr.POLYGON_FILL))
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL))
        viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        viewer.doc.addRenderer('COM projection', yr.PointsRenderer(rd_COM))

        # viewer.doc.addRenderer('x_axis', yr.VectorsRenderer(p2_x, p1_x, (0,0,255)))
        # viewer.doc.addRenderer('p1', yr.PointsRenderer(p1_x, (0, 0, 255)))
        # viewer.doc.addRenderer('p2', yr.PointsRenderer(p2_x, (0, 255, 0)))

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

        bvh_qs.append(env.skel.q)

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
        # p1_x.append(np.array([0., 0., 0.]))
        # p2_x.append(np.array([5., 0., 5.]))

        # front_point = env.skel.body("h_blade_left").to_world((0.0216 + 0.104, -0.0216 - 0.027, 0.))
        # rear_point = env.skel.body("h_blade_left").to_world((0.0216 - 0.104, -0.0216 - 0.027, 0.0))
        # print('Front point: ', front_point)
        # print('Rear point: ', rear_point)
        # p1_x.append(front_point)
        # p2_x.append(rear_point)

        # print(env.skel.com())

        # print(env.skel.body('h_blade_left').to_world((0., 0., 0.)))

    viewer.setSimulateCallback(simulateCallback)
    viewer.setMaxFrame(cumulative_frame_duration[-1]-1)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()

    # skelqs2bvh(bvh_file_name, env.skel, bvh_qs)


if __name__ == '__main__':
    main()
