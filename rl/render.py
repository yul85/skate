from fltk import Fl
import torch
from rl.ppo import PPO
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np

import pydart2 as pydart


def main():
    MOTION_ONLY = False

    pydart.init()

    env_name = 'spin'

    ppo = PPO(env_name, 1, visualize_only=True)
    # if not MOTION_ONLY:
    #     ppo.LoadModel('model/' + env_name + '.pt')
    ppo.env.Resets(False)
    ppo.env.ref_skel.set_positions(ppo.env.ref_motion[ppo.env.phase_frame])

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    dart_world = ppo.env.world
    viewer = hsv.hpSimpleViewer(rect=(0, 0, 1200, 800), viewForceWnd=False)
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ppo.env.ref_world, (150,150,255), yr.POLYGON_FILL))
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,240,255), yr.POLYGON_FILL))
        viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        viewer.doc.addRenderer('ground', yr.DartRenderer(ppo.env.ground, (255, 255, 255), yr.POLYGON_FILL))

    def postCallback(frame):
        ppo.env.ref_skel.set_positions(ppo.env.ref_motion[frame])

    def simulateCallback(frame):
        state = ppo.env.GetState(0)
        action_dist, _ = ppo.model(torch.tensor(state.reshape(1, -1)).float())
        action = action_dist.loc.detach().numpy()
        res = ppo.env.Steps(action)
        # res = ppo.env.Steps(np.zeros_like(action))
        # print(frame, ppo.env.ref_skel.current_frame, ppo.env.world.time()*ppo.env.ref_motion.fps)
        # print(frame, res[0][0])
        # if res[0][0] > 0.46:
        #     ppo.env.continue_from_now_by_phase(0.2)
        if res[2]:
            print(frame, 'Done')
            ppo.env.reset()

        # contact rendering

        del rd_contact_forces[:]
        del rd_contact_positions[:]

        if len(ppo.env.bodyIDs) != 0:
            # print(len(ppo.env.bodyIDs))
            for ii in range(len(ppo.env.contact_force)):
                # print(len(ppo.env.contact_force))
                if ppo.env.bodyIDs[ii] == 4:
                    body = ppo.env.skel.body('h_blade_left')
                else:
                    body = ppo.env.skel.body('h_blade_right')

                rd_contact_forces.append(ppo.env.contact_force[ii] / 100.)
                rd_contact_positions.append(body.to_world(ppo.env.contactPositionLocals[ii]))
        # contacts = ppo.env.world.collision_result.contacts
        # for contact in contacts:
        #     rd_contact_forces.append(contact.f/1000.)
        #     rd_contact_positions.append(contact.p)

    if MOTION_ONLY:
        viewer.setPostFrameCallback_Always(postCallback)
        viewer.setMaxFrame(len(ppo.env.ref_motion)-1)
    else:
        viewer.setSimulateCallback(simulateCallback)
        viewer.setMaxFrame(3000)

    # viewer.motionViewWnd.glWindow.planeHeight = -0.98 + 0.0251
    viewer.motionViewWnd.glWindow.planeHeight = 0.05
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()


if __name__ == '__main__':
    main()
