import os
from skate_ppo.hprun import main
import time

if __name__ == '__main__':
    argv = []

    env_id = 'jump'

    cur_time = time.strftime("%Y%m%d%H%M")

    os.environ["OPENAI_LOGDIR"] = os.getcwd() + '/' + env_id + '/log_' + cur_time
    os.environ["OPENAI_LOG_FORMAT"] = 'csv'
    argv.extend(['--env='+env_id])
    argv.extend(['--alg=ppo2'])
    argv.extend(['--num_env=8'])
    argv.extend(['--num_timesteps=2e7'])
    argv.extend(['--save_path='+env_id+'/'+'model_'+cur_time])
    argv.extend(['--num_hidden=64'])
    argv.extend(['--num_layers=2'])
    main(argv)
