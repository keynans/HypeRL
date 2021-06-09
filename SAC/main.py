import argparse
import pybullet_envs
import numpy as np
import torch
import time
import os
import socket

from models import SoftActorCritic
from helper import TimeFeatureWrapper
from models.utils import ZFilter, Identity

def set_mujoco():
    hostname = socket.gethostname()
    path = os.path.join(os.path.expanduser('~'),'.mujoco')
    name = 'mjkey.txt'
    if os.path.exists(os.path.join(path, name)):
        os.remove(os.path.join(path, name))
    os.symlink(os.path.join(path, f'{name}.{hostname}'), os.path.join(path, name))

set_mujoco()
import gym
import gym1

t0 = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#eval over mean action
def evaluate_agent(agent, eval_num, env_name, seed, total_timesteps, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            mean = agent.policy_network.sample_action(torch.Tensor(obs).view(1,-1).to(device), True)
            obs, reward, done, _ = eval_env.step(mean.detach().cpu().numpy()[0])
            avg_reward += reward

    avg_reward /= eval_episodes

    print ("Eval: %d  Total_timesteps: %d Evaluation: %f,  time-%d " % (eval_num, total_timesteps, avg_reward, int(time.time()-t0)))
    return avg_reward

def main(args):

    # Initialize environment and agent
    env = gym.make(args.env_name)
    
    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = {'grad_policy':[[]], 'grad_q1':[[]], 'policy_loss':[[]], 'q1_loss':[[]]}
    agent = SoftActorCritic(env.observation_space, env.action_space, args, logger)
    
    i = 0
    ep = 1
    evaluations = [evaluate_agent(agent,0, args.env_name,args.seed, i)] 
    if not args.no_hyper:
        hyperstr = 'Hyper_'
    else:
        hyperstr = ''
    file_name = "%s_%s_seed%s" % (hyperstr, args.env_name,args.seed)

    print("Start training...")

    while ep >= 1:
        episode_reward = 0
        state = env.reset()
        done = False
        j = 0
        
        while not done:
            # sample action
            if i > args.start_timesteps:
                action = agent.get_action(state)
            else:
                action = env.action_space.sample()
            
            if agent.replay_memory.get_len() > args.batch_size: 

                logger = agent.update_params(args.batch_size, args.gamma, args.tau, i)

            # prepare transition for replay memory push
            next_state, reward, done, _ = env.step(action)
            reward *= args.reward_scale
            i += 1
            j += 1
            episode_reward += reward

            ndone = 1 if j >= env._max_episode_steps else float(not done)
            agent.replay_memory.push((state,action,reward,next_state,ndone))
            state = next_state
        
            # eval episode
            if i % args.eval_freq == 0:
                evaluations.append(evaluate_agent(agent, i / args.eval_freq, args.env_name,args.seed, i))
                np.save("./results/%s" % (file_name), evaluations)
        
        if (agent.replay_memory.get_len() > args.batch_size) and (np.mean(logger['grad_policy'][-1]) > 1000): 
            print("grad_policy: {}".format(np.mean(logger['grad_policy'][-1])))
  
        if i >= args.max_time_steps:
            break
        
        logger['grad_policy'].append([])
        logger['grad_q1'].append([])
        ep += 1

        if args.debug and logger is not None:
            for key,value in logger.items():
                np.save("./gradient/"+ key +f"_{file_name}", np.array(value, dtype=object))
        


    np.save("./results/%s" % (file_name), evaluations)  

    env.close()


if __name__ == '__main__':

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="HopperBulletEnv-v0",
        help='name of the environment')
    parser.add_argument("--seed", default=0, type=int,
        help='seed')			
    parser.add_argument("--no_hyper", action="store_true", default=True)	# use regular critic
    parser.add_argument('--epsilon', type=float, default=1e-6,
        help='.....')
    parser.add_argument('--hidden_dim', type=int, default=256,
        help='regular hidden layers dim')
    parser.add_argument('--tau', type=float, default=0.005,
        help='soft update param')
    parser.add_argument('--lr', type=float, default=3e-4,
        help='regular lr')
    parser.add_argument('--hyper_lr', type=float, default=5e-5,
        help='hyper lr')
    parser.add_argument('--batch_size', type=int, default=256,
        help='batch_size')
    parser.add_argument('--replay_memory_size', type=int, default=1e6,
        help='replay memoery size')
    parser.add_argument('--max_time_steps', type=int, default=1e6,
        help='num of training time steps')
    parser.add_argument('--alpha', type=float, default=1.,
        help='max entropy vs. expected reward')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='discount factor')
    parser.add_argument("--start_timesteps", default=1e3, type=int,
        help='number of timesteps at the start for exploration')
    parser.add_argument('--min_log', type=float, default=-20,
        help='min log')
    parser.add_argument('--max_log', type=float, default=2,
        help='max log')
    parser.add_argument('--reward_scale', type=float, default=1,
        help='reward scale')
    parser.add_argument("--eval_freq", default=5e3, type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--entropy_tunning', type=bool, default=False)

    args = parser.parse_args()   

    args.device = device
    print(args)

    main(args)
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --env_name="HalfCheetah-v2" --alpha=0.2 --lr=1e-4 --start_timest4ps=10000
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --env_name="...-v2" --alpha=0.2 --lr=1e-5
