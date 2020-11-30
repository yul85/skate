import os
from fltk import Fl
import torch
from Examples.jump_various_speed.ppo_mp import PPO
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np

import pydart2 as pydart

from SkateUtils.DartMotionEdit import skelqs2bvh, DartSkelMotion


def main():
    MOTION_ONLY = False
    np.set_printoptions(precision=5)

    pydart.init()

    env_name = 'jump_various_speed'

    ppo = PPO(env_name, 0, visualize_only=True)
    if not MOTION_ONLY:
        if os.path.exists('model/'+env_name+'.pt'):
            ppo.LoadModel('model/' + env_name + '.pt')  # 06011454/1283.pt
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '1283' + '.pt')  # <= 2.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '1270' + '.pt') # <= 1.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '1122' + '.pt')  # <= 1.75
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '997' + '.pt')  # <= 1.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '940' + '.pt')  # <= 1.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '883' + '.pt')  # <= 1.5 >= 0.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '716' + '.pt')  # <= 1.75
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '573' + '.pt')  # <= 1.25
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '505' + '.pt')  # <= 2.0  >= 0.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '495' + '.pt')  # <= 1.25 >= 0.25
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '494' + '.pt')  # <= 1.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '425' + '.pt')  # <= 1.5  >= 0.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '412' + '.pt')  # <= 2.5  >= 0.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '409' + '.pt')  # <= 1.75  >= 0.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '403' + '.pt')  # <= 1.5  >= 0.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '389' + '.pt')  # <= 2.0  >= 0.5
        # ppo.LoadModel('jump_various_speed_model_201906011454/' + '386' + '.pt')  # <= 1.75
        # ppo.LoadModel('jump_various_speed_model_201906051533/' + '2864' + '.pt')  # <= 2.5

    ppo.env.flag_rsi(False)
    x_vel = [None]
    x_vel[0] = -1.5
    ppo.env.reset_with_x_vel(x_vel[0])
    # ppo.env.reset()
    ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(ppo.env.current_frame))

    # for bvh file
    bvh_qs = []
    bvh_file_name = 'ppo_jump_various_speed.bvh'
    skmo_file_name = 'ppo_jump_various_speed.skmo'
    motion = DartSkelMotion()

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    rd_COM = [None]
    rd_ext_force = [None]
    rd_ext_force_pos = [None]

    dart_world = ppo.env.world
    viewer_w, viewer_h = 1920, 1080
    viewer = hsv.hpSimpleViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55), viewForceWnd=False)
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ppo.env.ref_world, (194,207,245), yr.POLYGON_FILL))
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL))
        viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        viewer.doc.addRenderer('COM projection', yr.PointsRenderer(rd_COM))
        viewer.doc.addRenderer('ext force', yr.WideArrowRenderer(rd_ext_force, rd_ext_force_pos, lineWidth=.1, fromPoint=False))

    def postCallback(frame):
        ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(frame))

    def simulateCallback(frame):
        state = ppo.env.state()
        action_dist, _ = ppo.model(torch.tensor(state.reshape(1, -1)).float())
        action = action_dist.loc.detach().numpy()
        if False and frame == 15:
            ppo.env.ext_force = np.array([0., 0., 200.])
            ppo.env.ext_force_duration = 0.2
        res = ppo.env.step(action[0])

        q = [np.asarray(ppo.env.skel.q)]
        dq = [np.asarray(ppo.env.skel.dq)]

        # make skmo file
        motion.append(q[0], dq[0])
        # make bvh file
        bvh_qs.append(ppo.env.skel.q)

        if res[2]:
            print(frame, x_vel[0], 'Done')
            x_vel[0] -= 0.5
            # ppo.env.reset_with_x_vel(x_vel[0])

        # contact rendering
        contacts = ppo.env.world.collision_result.contacts
        del rd_contact_forces[:]
        del rd_contact_positions[:]
        for contact in contacts:
            if contact.skel_id1 == 0:
                rd_contact_forces.append(-contact.f/1000.)
            else:
                rd_contact_forces.append(contact.f/1000.)
            rd_contact_positions.append(contact.p)

        del rd_COM[:]
        com = ppo.env.skel.com()
        com[1] = 0.
        rd_COM.append(com)

        # ext force rendering
        del rd_ext_force[:]
        del rd_ext_force_pos[:]
        if ppo.env.ext_force_duration > 0.:
            rd_ext_force.append(ppo.env.ext_force/200.)
            rd_ext_force_pos.append(ppo.env.skel.body('h_pelvis').to_world())

    if MOTION_ONLY:
        viewer.setPostFrameCallback_Always(postCallback)
        viewer.setMaxFrame(len(ppo.env.ref_motion)-1)
    else:
        viewer.setSimulateCallback(simulateCallback)
        viewer.setMaxFrame(57)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()

    # skelqs2bvh(bvh_file_name, ppo.env.skel, bvh_qs)
    # motion.save(skmo_file_name)


if __name__ == '__main__':
    main()
