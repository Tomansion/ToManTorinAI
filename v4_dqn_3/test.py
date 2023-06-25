from env import Env
from agent import Agent
from time import sleep

import json

with open("config.json", "r") as f:
    conf = json.load(f)

model_name = conf["model_name"]


def test(
    delay=0.0,
    model_name=model_name,
    display=False,
    episodes=100,
    verbose=False,
    test=True,
):
    env = Env(test=test)
    nb_win = 0
    agent = Agent(env.get_state_size(), env.get_action_size(), model_name)

    agent.load(fail_if_not_found=True)

    if test:
        print("Testing on test env...")
    else:
        print("Testing on train env...")

    for i in range(episodes):
        # agent.reset_info()
        env.reset()

        if display:
            env.render()
            sleep(delay)

        while True:
            # get old state
            state, possible_actions = env.get_state()

            # === Move
            # perform move
            action_choice = agent.get_action(state, possible_actions, display=display)
            reward, game_done = env.move(action_choice)

            if display:
                print()
                env.render()
                if not game_done:
                    sleep(delay)

            if game_done:
                if reward > 0:
                    nb_win += 1
                break

        if verbose:
            # agent.print_info()
            print("Episode:", i + 1)
            print("Average win:", nb_win / (i + 1))

    print("Average score over", episodes, "episodes:", nb_win / episodes)
    env.stats()
    return nb_win / episodes


if __name__ == "__main__":
    # Profiling
    import pstats
    import cProfile

    # profiler = cProfile.Profile()
    # profiler.enable()

    # === Small model tests
    test(delay=3, model_name=model_name, display=True, episodes=1, test=True)
    # test(delay=0.6, display=True, episodes=1, test=False)
    # test(delay=0.6, display=True, episodes=1, test=True, best=True)
    # test(delay=0.6, display=True, episodes=1, test=False, best=True)

    # === Big model tests
    # test(episodes=1000, verbose=False, test=True)
    # test(episodes=1000, verbose=False, test=False)

    # === Test random
    # test_random(episodes=1000, verbose=False, test=True)
    # test_random(episodes=10000, verbose=False, test=False)
    # test_no_mistake(episodes=10000, verbose=True, test=True)
    # test_no_mistake(episodes=10000, verbose=True, test=False)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats(filename="stats.prof")
