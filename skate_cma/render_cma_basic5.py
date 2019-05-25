import os
from fltk import Fl
from skate_cma.skate_env2 import SkateDartEnv
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np
import pickle

from SkateUtils.DartMotionEdit import skelqs2bvh
import pydart2 as pydart


def main():
    MOTION_ONLY = False
    np.set_printoptions(precision=5)

    pydart.init()

    env_name = 'hmr_skating_basic5'
    max_time = 4.

    with open(env_name + '.skkey', 'rb') as skkey_file:
        skkey_states = pickle.load(skkey_file)

    # for bvh file
    bvh_qs = []
    bvh_file_name = 'render_cma.bvh'

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

    # file_path = 'jump_back_solution/xbest.skcma'
    file_path = 'hmr_skating_basic5_model_201905242151/xbest.skcma'
    try:
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
    except FileNotFoundError:
        print('file not found! just play keyframe')
        for i in range(len(state_duration)):
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
        bvh_qs.append(env.skel.q)

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

    # skelqs2bvh(bvh_file_name, env.skel, bvh_qs)


if __name__ == '__main__':
    main()
