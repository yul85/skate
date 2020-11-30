from fltk import Fl
import torch
from Examples.gliding_pd_action.ppo_mp import PPO
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np

import pydart2 as pydart

from SkateUtils.DartMotionEdit import DartSkelMotion
from matplotlib import pyplot as plt

def main():
    MOTION_ONLY = False
    PD_PLOT = False
    # PD_PLOT = True

    np.set_printoptions(precision=5)

    pydart.init()

    env_name = 'gliding'

    gain_p0 = []
    gain_p1 = []
    gain_p2 = []
    gain_p3 = []
    gain_p4 = []
    gain_p5 = []
    MA_DUR = 6
    ma_gain_p0 = []
    ma_gain_p1 = []
    ma_gain_p2 = []
    ma_gain_p3 = []
    ma_gain_p4 = []
    ma_gain_p5 = []

    ppo = PPO(env_name, 0, visualize_only=True)
    if not MOTION_ONLY:
        # ppo.LoadModel('gliding_model_202004061440/' + '282' + '.pt') # without kp info
        ppo.LoadModel('gliding_model_202004061641/' + '623' + '.pt')
        pass

    ppo.env.flag_rsi(False)
    ppo.env.reset()
    ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(ppo.env.current_frame))

    # for bvh file
    bvh_qs = []
    bvh_file_name = 'ppo_gliding_low_pd.bvh'
    skmo_file_name = 'ppo_gliding_low_pd.skmo'
    motion = DartSkelMotion()

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    rd_COM = [None]
    rd_ext_force = [None]
    rd_ext_force_pos = [None]
    rd_traj = []

    dart_world = ppo.env.world
    viewer_w, viewer_h = 1920, 1080
    # viewer_w, viewer_h = 1600, 900
    viewer = hsv.hpSimpleViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55), viewForceWnd=False)
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ppo.env.ref_world, (194,207,245), yr.POLYGON_FILL))
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL))
        viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        viewer.doc.addRenderer('COM projection', yr.PointsRenderer(rd_COM))
        viewer.doc.addRenderer('ext force', yr.WideArrowRenderer(rd_ext_force, rd_ext_force_pos, lineWidth=.1, fromPoint=False))
        viewer.doc.addRenderer('trajectory', yr.LinesRenderer(rd_traj))

    def postCallback(frame):
        ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(frame))

    def simulateCallback(frame):
        state = ppo.env.state()
        action_dist, _ = ppo.model(torch.tensor(state.reshape(1, -1)).float())
        action = action_dist.loc.detach().numpy()
        # if frame % 60 == 59:
        #     ppo.env.ext_force = np.array([0., 0., 400.])
        #     ppo.env.ext_force_duration = 0.1
        res = ppo.env.step(action[0])

        # for gain plotting
        if PD_PLOT:
            gain_p0.append(res[3]['kp'][ppo.env.skel.dof_index('j_thigh_left_x')])
            gain_p1.append(res[3]['kp'][ppo.env.skel.dof_index('j_shin_left_x')])
            gain_p2.append(res[3]['kp'][ppo.env.skel.dof_index('j_heel_left_x')])
            gain_p3.append(res[3]['kp'][ppo.env.skel.dof_index('j_thigh_right_x')])
            gain_p4.append(res[3]['kp'][ppo.env.skel.dof_index('j_shin_right_x')])
            gain_p5.append(res[3]['kp'][ppo.env.skel.dof_index('j_heel_right_x')])
            ma_gain_p0.append(sum(gain_p0[-MA_DUR:]) / MA_DUR if len(gain_p0) >= MA_DUR else sum(gain_p0) / len(gain_p0))
            ma_gain_p1.append(sum(gain_p1[-MA_DUR:]) / MA_DUR if len(gain_p1) >= MA_DUR else sum(gain_p1) / len(gain_p1))
            ma_gain_p2.append(sum(gain_p2[-MA_DUR:]) / MA_DUR if len(gain_p2) >= MA_DUR else sum(gain_p2) / len(gain_p2))
            ma_gain_p3.append(sum(gain_p3[-MA_DUR:]) / MA_DUR if len(gain_p3) >= MA_DUR else sum(gain_p3) / len(gain_p3))
            ma_gain_p4.append(sum(gain_p4[-MA_DUR:]) / MA_DUR if len(gain_p4) >= MA_DUR else sum(gain_p4) / len(gain_p4))
            ma_gain_p5.append(sum(gain_p5[-MA_DUR:]) / MA_DUR if len(gain_p5) >= MA_DUR else sum(gain_p5) / len(gain_p5))

        q = [np.asarray(ppo.env.skel.q)]
        dq = [np.asarray(ppo.env.skel.dq)]

        # make skmo file
        motion.append(q[0], dq[0])
        # make bvh file
        bvh_qs.append(ppo.env.skel.q)

        # if res[2]:
        #     print(frame, 'Done')
        #     ppo.env.reset()

        if PD_PLOT:
            fig = plt.figure(1)
            plt.clf()
            fig.add_subplot(6, 1, 1)
            # plt.axvspan(rfoot_contact_range[0], rfoot_contact_range[1], facecolor='0.5', alpha=0.3)
            plt.ylabel('left thigh')
            plt.plot(range(len(ma_gain_p0)), ma_gain_p0)
            fig.add_subplot(6, 1, 2)
            # plt.axvspan(rfoot_contact_range[0], rfoot_contact_range[1], facecolor='0.5', alpha=0.3)
            plt.ylabel('left knee')
            plt.plot(range(len(ma_gain_p1)), ma_gain_p1)
            fig.add_subplot(6, 1, 3)
            # plt.axvspan(rfoot_contact_range[0], rfoot_contact_range[1], facecolor='0.5', alpha=0.3)
            plt.ylabel('left ankle')
            plt.plot(range(len(ma_gain_p2)), ma_gain_p2)
            fig.add_subplot(6, 1, 4)
            # plt.axvspan(rfoot_contact_range[0], rfoot_contact_range[1], facecolor='0.5', alpha=0.3)
            plt.ylabel('right thigh')
            plt.plot(range(len(ma_gain_p3)), ma_gain_p3)
            fig.add_subplot(6, 1, 5)
            # plt.axvspan(rfoot_contact_range[0], rfoot_contact_range[1], facecolor='0.5', alpha=0.3)
            plt.ylabel('right knee')
            plt.plot(range(len(ma_gain_p4)), ma_gain_p4)
            fig.add_subplot(6, 1, 6)
            # plt.axvspan(rfoot_contact_range[0], rfoot_contact_range[1], facecolor='0.5', alpha=0.3)
            plt.ylabel('right ankle')
            plt.plot(range(len(ma_gain_p5)), ma_gain_p5)
            plt.show()
            plt.pause(0.001)

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

        # com rendering
        del rd_COM[:]
        com = ppo.env.skel.com()
        com[1] = 0.
        rd_COM.append(com)

        rd_traj.append(ppo.env.skel.com())

        # ext force rendering
        del rd_ext_force[:]
        del rd_ext_force_pos[:]
        if ppo.env.ext_force_duration > 0.:
            rd_ext_force.append(ppo.env.ext_force/500.)
            rd_ext_force_pos.append(ppo.env.skel.body('h_spine').to_world())

    if MOTION_ONLY:
        viewer.setPostFrameCallback_Always(postCallback)
        viewer.setMaxFrame(len(ppo.env.ref_motion)-1)
    else:
        viewer.setSimulateCallback(simulateCallback)
        viewer.setMaxFrame(1001)
        CAMERA_TRACKING = False
        if CAMERA_TRACKING:
            cameraTargets = [None] * (viewer.getMaxFrame()+1)

        def postFrameCallback_Always(frame):
            if CAMERA_TRACKING:
                if cameraTargets[frame] is None:
                    cameraTargets[frame] = ppo.env.skel.body(0).com()
                viewer.setCameraTarget(cameraTargets[frame])

        viewer.setPostFrameCallback_Always(postFrameCallback_Always)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()

    # skelqs2bvh(bvh_file_name, ppo.env.skel, bvh_qs)
    # motion.save(skmo_file_name)


if __name__ == '__main__':
    main()
