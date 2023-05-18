from train import define_parameters, run

if __name__ == "__main__":
    # Set options to activate or deactivate the game view, and its speed
    params = define_parameters()
    params["load_weights"] = True
    params["train"] = False
    print("Testing...")
    score = run(params)
    print(score)