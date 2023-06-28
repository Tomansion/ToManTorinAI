# Import the local gym environment
from env.santorini_env import Santorini
import json
from stable_baselines3 import DQN, PPO

with open("config.json", "r") as f:
    conf = json.load(f)

model_name = conf["model"]["name"]
nb_episodes = conf["train"]["episodes"]


print("Testing Santorinai")

env = Santorini(test=False, render_mode=None)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(progress_bar=True, log_interval=40, total_timesteps=nb_episodes * 50)
model.save(model_name)

model_loaded = PPO.load(model_name)

# Test the trained agent
env_test = Santorini(test=True, render_mode=None)
obs, _ = env_test.reset()

for i in range(10000):
    # print(f"Step {i}")
    action, _states = model_loaded.predict(obs, deterministic=True)
    action = int(action)
    # print(f"Action: {action}")
    obs, rewards, done, _, info = env_test.step(action)
    # print(f"Reward: {rewards}, Done: {done}, Info: {info}")

    env_test.render()
    if done:
        print(f"Step {i}")
        print("Goal reached!", "reward=", rewards)
        print("Info:", info)
        obs, _ = env_test.reset()
