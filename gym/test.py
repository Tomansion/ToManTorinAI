import json
from model_util import get_model

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

with open("config.json", "r") as f:
    conf = json.load(f)

model_name = conf["model"]["name"]
model, env = get_model(model_name)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env=env, n_eval_episodes=1000)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
print(env._get_info())

# Play a game with render
model, env = get_model(model_name, render=True)
obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, action_masks=env.action_masks())
    env.render()
    obs, rewards, done, _, info = env.step(action)

print(env._get_info())
