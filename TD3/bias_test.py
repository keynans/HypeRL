import socket
import os
import fcntl
import torch.multiprocessing as mp
import numpy as np

def lock_file(file):

    fo = open(file, "r+b")
    while True:
        try:
            fcntl.lockf(fo, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError:
            pass

    return fo

def release_file(fo):
    fcntl.lockf(fo, fcntl.LOCK_UN)
    fo.close()

def set_mujoco():
    hostname = socket.gethostname()
    path = os.path.join(os.path.expanduser('~'),'.mujoco')
    name = 'mjkey.txt'

    if os.path.exists(os.path.join(path, name)):
        os.remove(os.path.join(path, name))
    try:
        os.symlink(os.path.join(path, f'{name}.{hostname}'), os.path.join(path, name))
    except:
        os.remove(os.path.join(path, name))
        os.symlink(os.path.join(path, f'{name}.{hostname}'), os.path.join(path, name))
    return path, name

#path, name = set_mujoco()
#from mujoco_py import MjSimState
#import mujoco_py

#import pybullet_envs
import gym
import gym1
import torch
import argparse
import time
#import pybullet as p
from scipy.stats import pearsonr
from scipy import spatial
import copy

import torch.autograd as autograd
import math

import utils

t0 = time.time()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, total_timesteps, eval_num, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print ("Eval: %d  Total_timesteps: %d Evaluation: %f,  time-%d " % (eval_num, total_timesteps, avg_reward, int(time.time()-t0)))
	return avg_reward

def estimate_gradient(policy, critic, states, actions, ns):
	
    s = torch.FloatTensor(states).to(device)
    a = policy.actor(s)

    grad_a = autograd.Variable(a.data, requires_grad=True).to(device)
	
    qa = policy.critic.Q1(s, grad_a, None)
    
    grad = autograd.grad(outputs=qa, inputs=grad_a, grad_outputs=torch.FloatTensor(qa.size()).fill_(1.).to(device),
                              create_graph=False, retain_graph=False, only_inputs=True)[0]

    return grad.squeeze()

def estimate_real_grad(x, y):
	'''
    y: Nstate x Nsample
    x: Nstate x Nsample x Naction
    
    returns:
    q: Nstate, Naction
    '''
    
	Nstate, Nsample, Naction = x.shape
    
	y1 = y.unsqueeze(2).repeat(1, 1, Nsample)
	y2 = y.unsqueeze(1).repeat(1, Nsample, 1)
	delta = (y1 - y2).reshape(Nstate, Nsample * Nsample, 1)
    
	x1 = x.unsqueeze(2).repeat(1, 1, Nsample, 1)
	x2 = x.unsqueeze(1).repeat(1, Nsample, 1, 1)
	x_tilde = (x1 - x2).reshape(Nstate, Nsample * Nsample, Naction)

	pinv = torch.inverse(torch.bmm(x_tilde.transpose(1, 2) , x_tilde))
	pinv = torch.bmm(pinv, x_tilde.transpose(1, 2))
    
	q = torch.bmm(pinv, delta).squeeze(2)
	return q

def estimate_real_q_and_save(name ,env, policy, num_of_state, num_of_actions, t):
    real_q = []
    states = []
    actions = []

    for i in range(num_of_state):
        real_q.append([])
        actions.append([])
        steps = np.random.randint(200)
        j = 0
        state, done = env.reset(), False
        while j < steps:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            if done:
                state, done = env.reset(), False
            j += 1
            
        saved_state = env.env.sim.get_state().flatten()
        states.append(state)
        counter = env._elapsed_steps
        
        '''
        env.env._p.saveBullet("saved_states/state.bullet{}_{}_{}".format(name,t,i))
        x = env.unwrapped.robot.calc_state()
        policy_action = []
        for k in range(100):
            policy_action.append(policy.select_action(np.array(state)))
            state, reward1, done, _ = env.step(policy_action[-1])
            if done:
                state, done = env.reset(), False
			
        _elapsed =env._elapsed_steps
        base_state = state
	
        base_action = policy.select_action(np.array(base_state))
        state2, reward2, done, _ = env.step(base_action)
        states.append(base_state)
        '''
        base_action = policy.select_action(np.array(state))
        for j in range(num_of_actions):
            real_q[i].append([])
            print ("state {} action {} {}{}".format(i,j,name,t))
            noise =  (np.random.randn(base_action.shape[0]) * 0.1).clip(-0.5, 0.5)
            a = (base_action + noise).clip(-1,1)
            actions[i].append(a)
            '''
            state, done = env.reset(), False
            p.restoreState(fileName="saved_states/state.bullet{}_{}_{}".format(name,t,i))
            state_new = env.unwrapped.robot.calc_state()
            for k in range(100):
                state1, reward1, done, _ = env.step(policy_action[k])
                if done:
                    state1, done = env.reset(), False
			
            env._elapsed_steps = _elapsed
            '''
            state, done = env.reset(), False
            env.env.sim.set_state_from_flattened(saved_state)
            env._elapsed_steps = counter
            avg_reward = 0.
            discount = 1.
            state, reward, done, _ = env.step(a)
            avg_reward += (discount * reward)
            while not done:
                discount *= 0.99
                act = policy.select_action(np.array(state))
                state, reward, done, _ = env.step(act)
                avg_reward += (discount * reward)
            real_q[i][j].append(avg_reward)
		
        np.save("./q_plots/actions/actions_%s_%d" % (name,t), np.stack(actions,axis=0))
        np.save("./q_plots/q_data/real_q_%s_%d" % (name,t), np.array(real_q)) 
        np.save("./q_plots/states/states%s_%d" % (name,t), np.stack(states,axis=0))
    return np.array(real_q)

def get_state(env):
    return env.sim.get_state().flatten()

def set_state(s, env):
    env.reset()
    mj_state = MjSimState.from_flattened(s, env.sim)
    env.sim.set_state(mj_state)
    env.sim.forward()
    return env.env._get_obs()

def run_policy(env, get_action, num_steps):
	
    o, r, d, ep_steps, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_steps:
        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_steps += 1
        n += 1

        if d:
            ep_steps = 0
            o, r, d, ep_ret = env.reset(), 0, False, 0
            n += 1
		
    return env , ep_steps

def get_state_from_distribution(env, get_action):
    
    n = np.random.randint(1000)
    env, ep_steps = run_policy(env, get_action, n)

    return get_state(env), ep_steps
    
def get_states_from_distribution(env, get_action, n):
    states = []
    steps = []
    for i in range(n):
        obs , ep_steps = get_state_from_distribution(env, get_action)
        states.append(obs)
        steps.append(ep_steps)
        
    return np.stack(states), np.stack(steps)

def repeat_q_action(env, critic_target, get_action, real_state, s, init_a, ni, n_steps, a_repeat, discount=0.99, action_parameters=None):
    
    max_steps = 1000 - n_steps
    scores = []
    
    for i in range(ni):
        
        set_state(real_state, env)
        reward = 0.
        j = 0
        d = False
        a = init_a
        while not d and j < max_steps:
            o, r, d, _ = env.step(a)
            reward = reward + discount ** j * r
            j += 1

            # first 'a_repeat' steps will be the initial action
            if j >= a_repeat:
                a = get_action(o)
        
        if j == max_steps:
            with torch.no_grad():
                q1, q2 = critic_target(torch.FloatTensor(o).unsqueeze(0).to(device), torch.FloatTensor(a).unsqueeze(0).to(device))
                reward = reward + discount ** j * torch.min(q1, q2)[0].item()

        scores.append(reward)

    return np.array(scores)

def compare_qvals_and_real_vals(env, policy, ns=10, na=10, ni=10, std=0.3):
    
    real_states, ep_steps = get_states_from_distribution(env, policy.select_action, ns)
    all_actions = []
    all_scores = []

    for i in range(ns):
        print (i)
        actions = []
        scores = []

        real_state = real_states[i]
        n_steps = ep_steps[i]
        for j in range(na):
            set_state(real_state, env)
            action = policy.select_action(env.env._get_obs())
            actions.append(action)
            noise = np.random.normal(0, std, size=action.shape)
            a = (action + noise).clip(-1, 1)
            actions.append(a)
            r = repeat_q_action(env, policy.critic_target, policy.select_action, real_state, real_state, a, ni, n_steps, discount=0.99)
            scores.append(r)

        all_actions.append(actions)
        all_scores.append(scores)
        
    all_scores = np.array(all_scores)
    all_actions = np.array(all_actions)
    
    a = all_actions.reshape(ns * na, -1)
    a = torch.FloatTensor(a).to(device)

    obs_states = np.stack([set_state(s, env) for s in real_states])
    
    s = np.expand_dims(obs_states, axis=1)
    s = s.repeat(na, axis=1).reshape(ns * na, -1)
    s = torch.FloatTensor(s).to(device)

    qvals = policy.critic.Q1(s, a, None)
    qvals = qvals.reshape(ns, na, -1)

    mean_scores = all_scores#.mean(axis=2)
    qvals = qvals.data.cpu()
    
    results = {'real_states': real_states,
               'all_actions': all_actions,
               'obs_states': obs_states,
               'qvals': qvals,
               'mean_scores': mean_scores
              }
    
    return results

def iterate_actions(rank, return_dict, env, policy, real_state, na, ni, n_steps, std, a_repeat):
    
    actions = []
    scores = []
    
    set_state(real_state, env)     
    base_a = policy.select_action(env.env._get_obs())

    for j in range(na):
        noise = np.random.normal(0, std, size=base_a.shape) if j!=0 else np.zeros((base_a.shape))
        a = (base_a + noise).clip(-1, 1)
        actions.append(a)
        
        r = repeat_q_action(env, policy.critic_target, policy.select_action, real_state, real_state, a, ni, n_steps, a_repeat, discount=0.99, action_parameters=None)
        scores.append(r)
        return_dict[rank] = {'actions': actions, 'scores': scores}
    
    print(rank)

def compare_qvals_and_real_vals_parallel(env, policy, ns=10, na=10, ni=10, std=0.3, a_repeat=1, device='cuda', real_states=None):
    
    if real_states is None:
        real_states, ep_steps = get_states_from_distribution(env, policy.select_action, ns)
    
    processes = []
    envs = [copy.deepcopy(env) for _ in range(ns)]
        
    policy.actor.share_memory()
    manager = mp.Manager()
    return_dict = manager.dict()

    for rank in range(ns):
        p = mp.Process(target=iterate_actions, args=(rank, return_dict, envs[rank], policy, real_states[rank], na, ni, ep_steps[rank], std, a_repeat))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    all_actions = np.array([return_dict[i]['actions'] for i in range(ns)])
    all_scores = np.array([return_dict[i]['scores'] for i in range(ns)])

    a = all_actions.reshape(ns * na, -1)
    a = torch.FloatTensor(a).to(device)

    obs_states = np.stack([set_state(s, env) for s in real_states])
    
    s = np.expand_dims(obs_states, axis=1)
    s = s.repeat(na, axis=1).reshape(ns * na, -1)
    s = torch.FloatTensor(s).to(device)

    qvals = policy.critic.Q1(s, a, None)
    qvals = qvals.reshape(ns, na, -1)

    mean_scores = all_scores#.mean(axis=2)
    qvals = qvals.data.cpu().numpy()
    
    results = {'real_states': real_states,
               'all_actions': all_actions,
               'obs_states': obs_states,
               'qvals': qvals,
               'mean_scores': mean_scores
              }
    
    return results

def norm(vector):
    return np.sqrt(np.sum(x * x for x in vector))    

def cosine_similarity(vec_a, vec_b):
        norm_a = norm(vec_a)
        norm_b = norm(vec_b)
        dot = np.sum(a * b for a, b in zip(vec_a, vec_b))
        return dot / (norm_a * norm_b)

if __name__ == "__main__":
    
    dirr = './total_gradient'
    name = dirr + "/%s_10k_%d_%f.npy" % ("Hyper__MyAnt-v0_4", 3, 0.3)
    all_results = {}
    if os.path.exists(name):
        all_results = np.load(name ,allow_pickle=True).tolist()
        print("reload")
        els = list(all_results.keys())
        a = els[-1]
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HopperBulletEnv-v0")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument('--hyper_lr', type=float, default=5e-5) 	# hyper lr
    parser.add_argument('--policy_lr', type=float, default=3e-4) 	# policy lr
    parser.add_argument("--no_hyper", action="store_true")#,default=True)			# use regular critic
    parser.add_argument("--reverse_hyper", action="store_true")		# use reverse critic
    parser.add_argument("--random_hyper", action="store_true")		# use radom critic
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", type=float, default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--dir", default="./pytorch_models/")
    args = parser.parse_args()

    if not args.no_hyper:
        hyperstr = 'Hyper_'
        if args.reverse_hyper:
            hyperstr += '_Reverse'
        elif args.random_hyper:
            hyperstr += '_Random_to_g'

    else:
        hyperstr = ''
    file_name = f"{hyperstr}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {hyperstr}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])


    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "hyper": {"no_hyper":args.no_hyper,"reverse_hyper":args.reverse_hyper,"random_hyper":args.random_hyper},
        "hyper_lr": args.hyper_lr,
        "policy_noise" : args.policy_noise * max_action,
        "noise_clip" : args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "policy_lr": args.policy_lr
    }
    
    torch.multiprocessing.set_start_method('spawn')
    num_of_state = 15
    num_of_actions = 15
    num_of_iters = 1
    a_repeat = 3
    std = 0.3

    np.random.seed(args.seed)
    env.seed(args.seed)

    dirr = args.dir
    
    name = "./gradient_test/%s_10k_%d_%f.npy" % (file_name, a_repeat, std)
    all_results = {}
    start = 0
    if os.path.exists(name):
        all_results = np.load(name ,allow_pickle=True).tolist()
        print("reload")
        els = list(all_results.keys())
        start = els[-1] + 10000
        print(start)
        
    for t in range(start, 600000, 10000):

            policy = torch.load(dirr + "%s_%d" % (file_name,t),map_location=torch.device(device))
            #results = compare_qvals_and_real_vals(env, policy, num_of_state, num_of_actions, num_of_iters, std)
            results = compare_qvals_and_real_vals_parallel(env, policy, num_of_state, num_of_actions, num_of_iters, std, a_repeat, device, None)
        #    results1 = compare_qvals_and_real_vals_parallel(env, policy, num_of_state//2, num_of_actions, num_of_iters, std, a_repeat, device, None)
        #    results = np.load("./gradient_test/%s_%d_%d_%f.npy" % (file_name, t, a_repeat, std),allow_pickle=True).tolist()  
        #    qvals, obs_states, all_actions, real_states, mean_scores = [np.concatenate([results[k],results1[k]],axis=0)  for k in ['qvals', 'obs_states', 'all_actions', 'real_states', 'mean_scores']]
            
            qvals, obs_states, all_actions, real_states, mean_scores = [results[k] for k in ['qvals', 'obs_states', 'all_actions', 'real_states', 'mean_scores']]
            
            results['qvals'] = qvals
            results['obs_states'] = obs_states
            results['all_actions'] = all_actions
            results['real_states'] = real_states
            results['mean_scores'] = mean_scores

            all_results[t] = results
            np.save("./gradient_test/%s_10k_%d_%f.npy" % (file_name, a_repeat, std), all_results)

            est_grad = estimate_gradient(policy, policy.critic_target, obs_states, all_actions, num_of_state)
            real_grad = estimate_real_grad(torch.FloatTensor(all_actions).to(device), torch.FloatTensor(mean_scores.mean(axis=2)).to(device))
            results['est_grad'] = est_grad.cpu().numpy()
            results['real_grad'] = real_grad.cpu().numpy()

            all_results[t] = results
            np.save("./gradient_test/%s_10k_%d_%f.npy" % (file_name, a_repeat, std), all_results)