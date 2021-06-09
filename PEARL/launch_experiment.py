"""
Launcher for experiments with PEARL

"""
import os
import os.path as osp
import pathlib
import numpy as np
import gym
import argparse
import json
import torch
import socket

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def set_mujoco():
    hostname = socket.gethostname()
    path = os.path.join(os.path.expanduser('~'),'.mujoco')
    name = 'mjkey.txt'
    if os.path.exists(os.path.join(path, name)):
        os.remove(os.path.join(path, name))
    os.symlink(os.path.join(path, f'{name}.{hostname}'), os.path.join(path, name))

#set_mujoco()
from rlkit.envs.pybullet.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.hyper_policy import Hyper_Policy
from rlkit.torch.sac.hyper_q_network import Hyper_QNetwork
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.hyper_sac import HyperPEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.torch.sac.hyper_agent import HyperPEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config


def experiment(variant):

    # create multi-task environment
    env = NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_params']))
    
    #set seeds
    env.seed(variant['seed'])
    torch.manual_seed(variant['seed'])
    np.random.seed(variant['seed'])

    #sample tasks
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    use_pearl_context = variant['use_pearl_context']
    use_reverse_net = variant['use_reverse']
    use_hyper_net = variant['use_hyper']
    use_combine_net = variant['use_combine']
    
    hyperstr = ''
    if use_pearl_context:
        hyperstr += "Embtask_"
    if use_hyper_net:
        hyperstr += "Hyper_"
    if use_reverse_net:
        hyperstr +="reverse_"
    if use_combine_net:
        hyperstr +="combine_"

    print(hyperstr)

    file_name = "%s_%s_scale%d_seed%d" % (hyperstr, variant['env_name'],variant['algo_params']['reward_scale'],variant['seed'])
    latent_dim = variant['latent_size'] if use_pearl_context else variant['task_dim']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    network_model = Hyper_QNetwork if use_hyper_net else FlattenMlp
    policy_model = Hyper_Policy if use_hyper_net else TanhGaussianPolicy
    agent_algo = PEARLAgent if use_pearl_context else HyperPEARLAgent
    meta_algo = PEARLSoftActorCritic if use_pearl_context else HyperPEARLSoftActorCritic
    context_encoder = None

    if use_pearl_context:
        print ("pearl context")
        context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            use_reverse=use_reverse_net,
            use_combine=use_combine_net
        ).to(device)
    qf1 = network_model(
        hidden_sizes=[net_size, net_size, net_size],
        output_size=1,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        input_size=latent_dim+obs_dim+action_dim,
        z_dim=latent_dim,
        use_reverse=use_reverse_net,
        use_combine=use_combine_net
    ).to(device)
    qf2 = network_model(
        hidden_sizes=[net_size, net_size, net_size],
        output_size=1,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        input_size=latent_dim+obs_dim+action_dim,
        z_dim=latent_dim,
        use_reverse=use_reverse_net,
        use_combine=use_combine_net
    ).to(device)
    vf = network_model(
        hidden_sizes=[net_size, net_size, net_size],
        output_size=1,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=0,
        input_size=latent_dim+obs_dim,
        z_dim=latent_dim,
        use_reverse=use_reverse_net,
        use_combine=use_combine_net
    ).to(device)
    target_vf = network_model(
        hidden_sizes=[net_size, net_size, net_size],
        output_size=1,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=0,
        input_size=latent_dim+obs_dim,
        z_dim=latent_dim,
        use_reverse=use_reverse_net,
        use_combine=use_combine_net
    ).to(device)
    policy = policy_model(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        input_size=obs_dim+latent_dim,
        use_reverse=use_reverse_net,
        use_combine=use_combine_net
    ).to(device)
    agent = agent_algo(
        latent_dim,
        context_encoder,
        policy,
        use_hyper_net,
        **variant['algo_params']
    )
    algorithm = meta_algo(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf, target_vf],
        latent_dim=latent_dim,
        **variant['algo_params'],
        use_hyper_net = use_hyper_net,
        file_name=file_name
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()
    
def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def main(args):
    variant = default_config
    if args.config:
        with open(osp.join(args.config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
        variant['seed'] = args.seed
        variant['use_hyper'] = args.use_hyper
        variant['use_reverse'] = args.use_reverse
        variant['use_pearl_context'] = args.use_pearl_context
        variant['use_combine'] = args.use_combine
    experiment(variant)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PEARL')
    parser.add_argument('--config', type=str, default='configs/cheetah-vel-hard.json')
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--use_hyper", action="store_true",default=True)
    parser.add_argument("--use_reverse", action="store_true")
    parser.add_argument("--use_pearl_context", action="store_true",default=True)
    parser.add_argument("--use_combine", action="store_true")

    
    args = parser.parse_args()
    main(args)

