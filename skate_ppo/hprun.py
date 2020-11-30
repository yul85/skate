try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import os
import sys
import multiprocessing
import gym
import tensorflow as tf

from baselines.common.misc_util import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args
from baselines.common.tf_util import get_session
from baselines.bench import Monitor
from baselines.common import retro_wrappers
from baselines.common.wrappers import ClipActionsWrapper
from baselines import logger
from importlib import import_module
from skate_ppo.envs import get_env, get_env_parameter


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    env = get_env(env_id)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env


def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()

    def make_thunk(rank, _initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=_initializer
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, _initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, _initializer=None) for i in range(num_env)])


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(args.env, None, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

    return env


def train(args, extra_args):
    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_env_parameter(args.env)

    alg_kwargs.update(extra_args)

    env = build_env(args)
    alg_kwargs['network'] = 'mlp'

    # print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    """
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible

    :param args:
    :return:
    """
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)
    if args.save_path is not None and rank == 0:
        save_path = os.path.expanduser(args.save_path)
        model.save(save_path)

    env.close()


if __name__ == '__main__':
    """
      -h, --help            show this help message and exit
      --env ENV             environment ID (default: Reacher-v2)
      --env_type ENV_TYPE   type of environment, used when the environment type
                            cannot be automatically determined (default: None)
      --seed SEED           RNG seed (default: None)
      --alg ALG             Algorithm (default: ppo2)
      --num_timesteps NUM_TIMESTEPS
      --network NETWORK     network type (mlp, cnn, lstm, cnn_lstm, conv_only)
                            (default: None)
      --gamestate GAMESTATE
                            game state to load (so far only used in retro games)
                            (default: None)
      --num_env NUM_ENV     Number of environment copies being run in parallel.
                            When not specified, set to number of cpus for Atari,
                            and to 1 for Mujoco (default: None)
      --reward_scale REWARD_SCALE
                            Reward scale factor. Default: 1.0 (default: 1.0)
      --save_path SAVE_PATH
                            Path to save trained model to (default: None)
      --save_video_interval SAVE_VIDEO_INTERVAL
                            Save video every x steps (0 = disabled) (default: 0)
      --save_video_length SAVE_VIDEO_LENGTH
                            Length of recorded video. Default: 200 (default: 200)
      --play

      --alg=ppo2
      --env=Humanoid-v2
      --network=mlp
      --num_timesteps=2e7
      --ent_coef=0.1
      --num_hidden=32
      --num_layers=3
      --value_network=copy 
      --load_path=~/models/pong_20M_ppo2
      
      OPENAI_LOGDIR=$HOME/logs/cartpole-ppo
      OPENAI_LOG_FORMAT=csv
    """
    import time
    argv = []

    env_id = 'speed_skating'

    cur_time = time.strftime("%Y%m%d%H%M")

    os.environ["OPENAI_LOGDIR"] = os.getcwd() + '/' + env_id + '/log_' + cur_time
    os.environ["OPENAI_LOG_FORMAT"] = 'csv'
    argv.extend(['--env='+env_id])
    argv.extend(['--alg=ppo2'])
    argv.extend(['--num_env=8'])
    argv.extend(['--num_timesteps=1e8'])
    argv.extend(['--save_path='+env_id+'/'+'model_'+cur_time])
    argv.extend(['--num_hidden=64'])
    argv.extend(['--num_layers=2'])
    main(argv)
