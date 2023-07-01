from env import Santorini
import json

from stable_baselines3 import PPO

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

with open("config.json", "r") as f:
    conf = json.load(f)

model_name = conf["model"]["name"]
nb_episodes = conf["train"]["episodes"]


total_timesteps = nb_episodes

env = Santorini()
model = MaskablePPO(
    "MlpPolicy", env, verbose=1, tensorboard_log="./logs/", learning_rate=0.0001
)
model.learn(
    total_timesteps,
    progress_bar=True,
    tb_log_name=model_name,
    reset_num_timesteps=False,
)

# Save the trained model
model.save(model_name)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env=env, n_eval_episodes=1000)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
