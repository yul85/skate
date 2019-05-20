from fltk import Fl
import torch
from Examples.jump.ppo_mp import PPO
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np

import pydart2 as pydart


def main():
    MOTION_ONLY = False
    np.set_printoptions(precision=5)

    pydart.init()

    env_name = 'jump'

    ppo = PPO(env_name, 0, visualize_only=True)
    if not MOTION_ONLY:
        # ppo.LoadModel('jump_model_201905091411/' + '1070' + '.pt')
        # if os.path.exists('model/'+env_name+'.pt'):
        #     ppo.LoadModel('model/' + env_name + '.pt')
        ppo.LoadModel('model/' + 'param' + '.pt')
        pass

    ppo.env.Resets(False)
    ppo.env.ref_skel.set_positions(ppo.env.ref_motion.qs[ppo.env.phase_frame])

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    rd_COM = [None]
    dart_world = ppo.env.world
    viewer_w, viewer_h = 1280, 720
    viewer = hsv.hpSimpleViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55), viewForceWnd=False)
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ppo.env.ref_world, (194,207,245), yr.POLYGON_FILL))
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL))
        viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        viewer.doc.addRenderer('COM projection', yr.PointsRenderer(rd_COM))

    def postCallback(frame):
        ppo.env.ref_skel.set_positions(ppo.env.ref_motion.qs[frame])

    def simulateCallback(frame):
        state = ppo.env.GetState(0)
        action_dist, _ = ppo.model(torch.tensor(state.reshape(1, -1)).float())
        action = action_dist.loc.detach().numpy()
        # res = ppo.env.Steps(action)
        res = ppo.env.Steps(np.zeros_like(action))

        if res[2]:
            print(frame, 'Done')
            ppo.env.reset()

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

    if MOTION_ONLY:
        viewer.setPostFrameCallback_Always(postCallback)
        viewer.setMaxFrame(len(ppo.env.ref_motion)-1)
    else:
        viewer.setSimulateCallback(simulateCallback)
        viewer.setMaxFrame(3000)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()


if __name__ == '__main__':
    main()
