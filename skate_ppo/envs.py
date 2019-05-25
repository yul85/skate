from skate_ppo.speed_skating.skate_env_using_ref import SkateDartEnv as SpeedSkatingEnv
from skate_ppo.jump.skate_env_using_ref import SkateDartEnv as JumpEnv


def get_env(env_id):
    if env_id == 'speed_skating':
        return SpeedSkatingEnv()
    elif env_id == 'jump':
        return JumpEnv()


def get_env_parameter(env_id):
    """
        Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)
        Parameters:
        ----------
        network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                          specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                          tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                          neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                          See common/models.py/lstm for more details on using recurrent nets in policies

        env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                          The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.

        nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                          nenv is number of environment copies simulated in parallel)

        total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

        ent_coef: float                   policy entropy coefficient in the optimization objective

        lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                          training and 0 is the end of the training.
        vf_coef: float                    value function loss coefficient in the optimization objective

        max_grad_norm: float or None      gradient norm clipping coefficient

        gamma: float                      discounting factor

        lam: float                        advantage estimation discounting factor (lambda in the paper)

        log_interval: int                 number of timesteps between logging events

        nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                          should be smaller or equal than number of environments run in parallel.

        noptepochs: int                   number of training epochs per update

        cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                          and 0 is the end of the training
        save_interval: int                number of timesteps between saving events

        load_path: str                    path to load the model from

        **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                          For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    """
    # # deepmimic style
    # return dict(
    #     nsteps=4096,
    #     nminibatches=256,
    #     lam=0.95,
    #     gamma=0.99,
    #     noptepochs=1,
    #     log_interval=1,
    #     save_interval=1,
    #     ent_coef=0.0,
    #     lr=lambda f: 5e-5 * f,
    #     cliprange=0.2,
    #     value_network='copy'
    # )

    # openai style
    # return dict(
    #     nsteps=256,
    #     nminibatches=32,
    #     lam=0.95,
    #     gamma=0.99,
    #     noptepochs=10,
    #     log_interval=1,
    #     save_interval=1,
    #     ent_coef=0.0,
    #     lr=lambda f: 3e-4 * f,
    #     cliprange=0.2,
    #     value_network='copy'
    # )

    # last style
    return dict(
        nsteps=256,
        nminibatches=128,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        save_interval=10,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )
