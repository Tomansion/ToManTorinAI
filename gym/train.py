from env import GymnasiumEnv

from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker

total_timesteps = 10000

env = GymnasiumEnv()
# env = ActionMasker(env, get_action_masks)  # Wrap to enable masking

model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
# model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps, progress_bar=True)

# Save the trained model
model.save("ppo_gymnasium")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env=env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
