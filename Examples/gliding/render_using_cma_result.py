from fltk import Fl
import torch
from Examples.gliding.ppo_mp import PPO
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np

import pydart2 as pydart

from SkateUtils.DartMotionEdit import skelqs2bvh, DartSkelMotion


def main():
    MOTION_ONLY = False
    np.set_printoptions(precision=5)

    pydart.init()

    env_name = 'gliding'

    ppo = PPO(env_name, 0, visualize_only=True)
    if not MOTION_ONLY:
        ppo.LoadModel('gliding_model_201908081441/' + '728' + '.pt')
        # ppo.LoadModel('gliding_model_201908081441/' + '165' + '.pt')  # first endless example
        # ppo.LoadModel('crossover_fast_low_pd_ref_model_201906041148/' + '433' + '.pt')
        # ppo.LoadModel('crossover_fast_low_pd_ref_model_201906041148/' + '471' + '.pt')
        # ppo.LoadModel('crossover_fast_low_pd_ref_model_201906041148/' + '484' + '.pt')  # draw circle example
        # ppo.LoadModel('crossover_fast_low_pd_ref_model_201906041148/' + '723' + '.pt')  #
        # ppo.LoadModel('model/' + 'param' + '.pt')
        # if os.path.exists('model/'+env_name+'.pt'):
        #     ppo.LoadModel('model/' + env_name + '.pt')
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

        q = [np.asarray(ppo.env.skel.q)]
        dq = [np.asarray(ppo.env.skel.dq)]

        # make skmo file
        motion.append(q[0], dq[0])
        # make bvh file
        bvh_qs.append(ppo.env.skel.q)

        # if res[2]:
        #     print(frame, 'Done')
        #     ppo.env.reset()

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
        viewer.setMaxFrame(301)
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
