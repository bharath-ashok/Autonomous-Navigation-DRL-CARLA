'''
Terminology:
	Episode: one go of the car trying to "live" in the simulation and earn max rewards.
				Episode start from spawning a new car and ends with either car crashing or episode duration limit running out

	Timestep: one frame through simulation: the car gets a obs, reward from prior step and then it makes a decision on control input and sends it to simulation

	Reward logic: each timestep a logic is applied to calculate a reward from latest step. 
				This logic represents you describing the desired behaviour the car needs to learn.

	Policy/model: our objective, what we need learn as part of RL. This is the latest set of rules on what to do at a given camera image.

	Iterations: RL training sessions (multiple episodes and timesteps) when a policy/model is saved. So the policy is changed throughout one iteration
				but then saved in a new file at the end of iteration. This allows to test all models later at different stages of training  
'''	

from stable_baselines3 import PPO #PPO
from stable_baselines3.common.env_checker import check_env

import os
from new_env import CarEnv
import time
from stable_baselines3.common.callbacks import EvalCallback

print('This is the start of training script')

print('setting folders for logs and models')
models_dir = f"models/1811/{int(time.time())}/"
logdir = f"logs/1811/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('connecting to env..')

env = CarEnv()  # Try check_env(env, warn=True) 
check_env(env, warn=True) 
env.reset()

print('Env has been reset as part of launch')

model = PPO(
    'MlpPolicy', 
    env, 
	
    verbose=1, 
    learning_rate=0.00003, #0.00003
    n_steps=1024, 
    batch_size=64, #128
    clip_range=0.1, 
    gamma=0.99, 
    normalize_advantage=True,
    tensorboard_log=logdir
)

TIMESTEPS = 2000000 
iters = 0 # how long is each training iteration - individual steps
while iters < 1:  # how many training iterations you want 
	iters += 1
	print('Iteration ', iters,' is to commence...')
	# Create the callback: check every 10000 steps
	eval_callback = EvalCallback(env, best_model_save_path=models_dir,
								 log_path=logdir, eval_freq=10000,
								 deterministic=True, render=False)

	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=eval_callback)
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")

env.close()
	