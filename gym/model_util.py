import json, os

from sb3_contrib import MaskablePPO


def get_model(model_name=None, render=False, load_enemy=True):
    from env import Santorini_1, Santorini_2, Santorini_3, Santorini_4

    if model_name is None:
        # Load from config
        with open("config.json", "r") as f:
            conf = json.load(f)

        model_policy = conf["model"]["policy"]
        model_env = conf["model"]["env"]
        model_name = conf["model"]["name"]

    else:
        # Load from model_name
        with open("models/" + model_name + "/config.json", "r") as f:
            conf = json.load(f)

        model_policy = conf["policy"]
        model_env = conf["env"]

    if model_env == 1:
        env = Santorini_1(render, load_enemy)
    elif model_env == 2:
        env = Santorini_2(render, load_enemy)
    elif model_env == 3:
        env = Santorini_3(render, load_enemy)
    elif model_env == 4:
        env = Santorini_4(render, load_enemy)
    else:
        raise ValueError("Invalid env")

    model = MaskablePPO(
        model_policy,
        env,
        verbose=1,
        learning_rate=0.0001,
        
    )
    # tensorboard_log="./logs/",

    # Add parameters if model exists
    if os.path.exists("models/" + model_name + "/" + model_name + ".zip"):
        model.set_parameters(
            "models/" + model_name + "/" + model_name, exact_match=True
        )

    return model, env
