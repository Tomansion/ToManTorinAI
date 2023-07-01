import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env import GymnasiumEnv

# Define your gymnasium environment
env = GymnasiumEnv()

model = PPO.load("ppo_gymnasium")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env=env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
