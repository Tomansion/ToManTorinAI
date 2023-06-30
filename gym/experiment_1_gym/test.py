# Import the local gym environment
from env.santorini_env import Santorini
import json
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib import MaskablePPO
from time import sleep

with open("config.json", "r") as f:
    conf = json.load(f)

model_name = conf["model"]["name"]
nb_episodes = conf["train"]["episodes"]

print("Testing Santorinai")

# Test the trained agent
# env_test = Santorini(test=True, render_mode="emoticons")
env_test = Santorini(test=True, render_mode=None)

if "dqn" in model_name:
    model = DQN.load(model_name)
elif "ppo" in model_name:
    model = MaskablePPO.load(model_name)
elif "a2c" in model_name:
    model = A2C.load(model_name)
else:
    raise ValueError(f"Unknown algorithm")

obs, _ = env_test.reset()


nb_episodes = 1000
episodes = 0
while episodes < nb_episodes:
    # env_test.render()

    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    
    # sleep(2)
    
    obs, rewards, done, _, info = env_test.step(action)

    if done:
        print(f"Step {episodes} / {nb_episodes}")
        print("Info:", info)
        obs, _ = env_test.reset()
        episodes += 1

nb_win = info["nb_win"]
win_rate = int(nb_win / nb_episodes * 100)
print(f"Win rate: {win_rate}%")
