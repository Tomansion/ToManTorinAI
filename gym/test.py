# Import the local gym environment
from env.santorini_env import Santorini
import json
from stable_baselines3 import DQN, PPO
from time import sleep

with open("config.json", "r") as f:
    conf = json.load(f)

model_name = conf["model"]["name"]

print("Testing Santorinai")

model_loaded = PPO.load(model_name)

# Test the trained agent
# env_test = Santorini(test=True, render_mode="emoticons")
env_test = Santorini(test=True, render_mode=None)
obs, _ = env_test.reset()


nb_episodes = 1000
episodes = 0
while episodes < nb_episodes:
    # print(f"Step {i}")
    action, _states = model_loaded.predict(obs, deterministic=True)
    action = int(action)
    # print(f"Action: {action}")
    env_test.render()
    sleep(2)
    obs, rewards, done, _, info = env_test.step(action)
    # print(f"Reward: {rewards}, Done: {done}, Info: {info}")

    if done:
        print(f"Step {episodes} / {nb_episodes}")
        print("Info:", info)
        obs, _ = env_test.reset()
        episodes += 1

nb_win = info["nb_win"]
win_rate = int(nb_win / nb_episodes * 100)
print(f"Win rate: {win_rate}%")