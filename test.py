from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
import time
from environment import CarEnv

# Directory containing saved models
models_dir = "models/2110//1729498948"
logdir = "logs/2110//1729498948"

# Ensure the directories exist
assert os.path.exists(models_dir), f"Model directory {models_dir} does not exist!"
assert os.path.exists(logdir), f"Log directory {logdir} does not exist!"

# List all the saved models in the directory
model_files = [f for f in sorted(os.listdir(models_dir)) if f.endswith('.zip')]
assert len(model_files) > 0, "No models found in the models directory!"

# Load the environment
env = CarEnv()
check_env(env)
env.reset()

# Select the most recent model to test
latest_model_path = os.path.join(models_dir, model_files[-1])
print(f"Loading model from {latest_model_path}")
model = PPO.load(latest_model_path, env=env)

# Number of episodes to test
num_test_episodes = 5

for episode in range(num_test_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    timestep = 0

    print(f"Starting Episode {episode + 1}")

    while not done:
        # Get the action from the model
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        timestep += 1
        if done or truncated:
            break

    print(f"Episode {episode + 1} finished. Total Reward: {total_reward} after {timestep} timesteps.\n")

print("Testing completed.")
env.close()
