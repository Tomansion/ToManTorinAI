import json, os

from model_util import get_model

from sb3_contrib.common.maskable.evaluation import evaluate_policy

with open("config.json", "r") as f:
    conf = json.load(f)

model_name = conf["model"]["name"]
model_policy = conf["model"]["policy"]
nb_episodes = conf["train"]["episodes"]
model_env = conf["model"]["env"]

model, env = get_model()

# Train the model

ep_nb = 0
while ep_nb < nb_episodes:
    print("Episodes: " + str(ep_nb) + "/" + str(nb_episodes))
    model.learn(
        10000,
        progress_bar=True,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )

    ep_nb += 10000

    # Save the trained model
    print("Saving model")
    os.makedirs("models/" + model_name, exist_ok=True)
    model.save("models/" + model_name + "/" + model_name)
    with open("models/" + model_name + "/config.json", "w") as f:
        json.dump({"policy": model_policy, "env": model_env}, f)

    # Test the trained model
    print("Evaluating model")
    mean_reward, std_reward = evaluate_policy(model, env=env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    print(env._get_info())
