import csv
import os
from environment import CarEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Directories
model_id = "XXXXXXXX"   # Update this as needed   1734705949
models_dir = f"{model_id}"
logdir = f"{model_id}"
step_log_file = os.path.join(logdir, "step_results_town02.csv")

# Ensure the directories exist
assert os.path.exists(models_dir), f"Model directory {models_dir} does not exist!"
assert os.path.exists(logdir), f"Log directory {logdir} does not exist!"

# Log file for step-wise results

# Prepare the environment
env = CarEnv()
check_env(env)
env.reset()

# List all saved models
model_files = [f for f in sorted(os.listdir(models_dir)) if f.endswith('.zip')]
assert len(model_files) > 0, "No models found in the models directory!"

# Load the best or most recent model
best_model_file = ('best_model' if 'best_model' in model_files else model_files[-1])
latest_model_path = os.path.join(models_dir, best_model_file)
print(f"Loading model from {latest_model_path}")
model = PPO.load(latest_model_path, env=env)

# Number of test episodes
num_test_episodes = 10

# Prepare to log results
fields = ["Episode", "Step", "Action", "Reward", "Total Reward", "Done", "Truncated"]
if not os.path.exists(step_log_file):
    with open(step_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)

# Testing loop
for episode in range(1, num_test_episodes + 1):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0

    print(f"Starting Episode {episode}")

    while not done and not truncated:
        # Get the action from the model
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Log the step data
        # with open(step_log_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([episode, step, action, reward, total_reward, done, truncated])

    print(f"Episode {episode} finished. Total Reward: {total_reward} after {step} steps.\n")

env.close()
