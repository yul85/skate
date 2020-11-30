from fltk import Fl
import numpy as np
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from baselines.common.cmd_util import common_arg_parser
from baselines.common.vec_env import VecEnv

from skate_ppo.hprun import parse_cmdline_kwargs, train


def main(args):
    MPI = None
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        # logger.configure()
    else:
        # logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)
    render_env = env.envs[0]
    render_env.flag_noise(False)
    render_env.flag_rsi(False)

    MOTION_ONLY = False
    np.set_printoptions(precision=5)

    render_env.ref_skel.set_positions(render_env.ref_motion.qs[0])

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    rd_COM = [None]
    dart_world = render_env.world
    viewer_w, viewer_h = 1280, 720
    viewer = hsv.hpSimpleViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55), viewForceWnd=False)
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(render_env.ref_world, (194,207,245), yr.POLYGON_FILL))
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL))
        viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        viewer.doc.addRenderer('COM projection', yr.PointsRenderer(rd_COM))

    obs = [None]
    state = [None]
    episode_rew = [None]

    obs[0] = env.reset()
    state[0] = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))
    episode_rew[0] = 0.

    def def_action_loc():
        self = model.act_model
        self.action_loc = self.pd.mode()

    def step2(observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)
        Parameters:
        ----------
        observation     observation data (either single or a batch)
        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        self = model.act_model

        a, v, _state, neglogp = self._evaluate([self.action_loc, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if _state.size == 0:
            _state = None
        return a, v, _state, neglogp

    def_action_loc()
    model.act_model.step2 = step2
    model.step2 = model.act_model.step2

    def postCallback(frame):
        render_env.ref_skel.set_positions(render_env.ref_motion.qs[frame])

    def simulateCallback(frame):
        if state[0] is not None:
            actions, _, state[0], _ = model.step2(obs[0], S=state[0], M=dones)
        else:
            actions, _, _, _ = model.step2(obs[0])

        obs[0], rew, done, _ = env.step(actions)
        episode_rew[0] += rew[0] if isinstance(env, VecEnv) else rew
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew[0] = 0.
            obs[0] = env.reset()

        # contact rendering
        contacts = render_env.world.collision_result.contacts
        del rd_contact_forces[:]
        del rd_contact_positions[:]
        for contact in contacts:
            if contact.skel_id1 == 0:
                rd_contact_forces.append(-contact.f/1000.)
            else:
                rd_contact_forces.append(contact.f/1000.)
            rd_contact_positions.append(contact.p)

        del rd_COM[:]
        com = render_env.skel.com()
        com[1] = 0.
        rd_COM.append(com)

    if MOTION_ONLY:
        viewer.setPostFrameCallback_Always(postCallback)
        viewer.setMaxFrame(len(render_env.ref_motion)-1)
    else:
        viewer.setSimulateCallback(simulateCallback)
        viewer.setMaxFrame(3000)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()


if __name__ == '__main__':
    import os
    argv = []

    env_id = 'jump'
    load_path = env_id+'/log_201905242139/checkpoints/03000'
    # env_id = 'speed_skating'
    # load_path = env_id+'/log_201905242139/checkpoints/03610'

    os.environ["OPENAI_LOGDIR"] = os.getcwd() + '/'+env_id+'/log_render'
    os.environ["OPENAI_LOG_FORMAT"] = 'stdout'
    argv.extend(['--env='+env_id])
    argv.extend(['--alg=ppo2'])
    argv.extend(['--num_env=1'])
    argv.extend(['--num_timesteps=0'])
    argv.extend(['--load_path='+load_path])
    main(argv)
