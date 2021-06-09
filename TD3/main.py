import numpy as np
import torch
import pybullet_envs
import argparse
import os
import time
import socket

from normalizer import normalizer
import utils
import TD3
import OurDDPG
import DDPG

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

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, total_timesteps, eval_num, obs_normalizer, eval_episodes=10):
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


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="AntBulletEnv-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument('--hyper_lr', type=float, default=5e-5) 	# hyper lr
	parser.add_argument('--policy_lr', type=float, default=3e-4) 	# policy lr
	parser.add_argument("--no_hyper", action="store_true")					# use regular critic
	parser.add_argument("--reverse_hyper", action="store_true")		# use reverse critic
	parser.add_argument("--random_hyper", action="store_true")		# use radom critic
	parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
	parser.add_argument("--policy_noise", type=float, default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--normalize_obs", action="store_true")		# normalize the observations
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	if not args.no_hyper:
		hyperstr = 'Hyper_'
		if args.reverse_hyper:
			hyperstr += '_Reverse'
		elif args.random_hyper:
			hyperstr += '_Random'

	else:
		hyperstr = 'D2RL_'


	file_name = f"{hyperstr}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {hyperstr}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./pytorch_models/" + args.env):
		os.makedirs("./pytorch_models/" + args.env)

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
	logger = {'q1':[[]],'loss1':[[]],'critic':[[]],'policy':[[]]}

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
		"policy_lr": args.policy_lr,
		"logger" : logger
	}
	
	print (kwargs)
	# Initialize policy
	policy = TD3.TD3(**kwargs)

	obs_normalizer = normalizer(state_dim, do_normalize=args.normalize_obs)
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, obs_normalizer)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed, 0, 0, obs_normalizer)]

	state, done = env.reset(), False

	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			nois = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			action = (
				policy.select_action(np.array(state))
				+ nois
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store un normalize data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			logger = policy.train(replay_buffer, args.batch_size, args.debug)

		if done: 					
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
			if args.debug:
				for key,value in logger.items():
					print("{} {}".format(key, np.mean(value[-1])))

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed, t+1, (t + 1) // args.eval_freq, obs_normalizer))
			np.save(f"./results/{file_name}", evaluations)

			# save gradient
			if args.debug and logger is not None:
				for key,value in logger.items():
					np.save("./grad/"+ key +f"_{file_name}", np.array(value, dtype=object))

	np.save("./results/%s" % (file_name), evaluations)  
