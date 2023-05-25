from train import define_parameters, run

if __name__ == "__main__":
    # Set options to activate or deactivate the game view, and its speed
    params = define_parameters()
    params["load_weights"] = True
    params["train"] = False
    params["episodes"] = 100
    params["delay_between_moves"] = 0
    print("Testing...")
    score = run(params)
    print("Final average score:", score)
