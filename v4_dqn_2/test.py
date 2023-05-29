from env import Env
from agent import Agent
from time import sleep


def test(delay=0.0, display=False, episodes=100, verbose=False, test=True, best=False):
    env = Env(test=test)

    if best:
        move_agent = Agent(env.get_state_size(), env.get_action_size(), "best_move_agent")
        build_agent = Agent(env.get_state_size(), env.get_action_size(), "best_build_agent")
    else:
        move_agent = Agent(env.get_state_size(), env.get_action_size(), "move_agent")
        build_agent = Agent(env.get_state_size(), env.get_action_size(), "build_agent")

    move_agent.load()
    build_agent.load()

    score_total = 0
    
    if test:
        print("Testing on test env...")
    else:
        print("Testing on train env...")

    for i in range(episodes):
        env.reset()

        if display:
            env.render()
            sleep(delay)

        while True:
            move_done, build_done = False, False
            # get old state
            move_state_old = env.get_move_state()

            # === Move
            # perform move
            move_choice = move_agent.get_action(move_state_old)
            move_reward, move_done = env.move(move_choice)
            if display:
                print()
                print("Move choice:", move_choice)
                env.render()
                print("Move reward:", move_reward)
                print("Move done:", move_done)
                print()
                if not move_done:
                    sleep(delay)

            if not move_done:
                build_state_after_move = env.get_build_state()
                # === Build
                # perform build
                build_choice = build_agent.get_action(build_state_after_move)
                build_reward, build_done = env.build(build_choice)
                if display:
                    print("Build choice:", build_choice)
                    env.render()
                    print("Build reward:", build_reward)
                    print("Build done:", build_done)
                    print()
                    if not build_done:
                        sleep(delay)

            if move_done or build_done:
                break

            # Move enemy
            enemy_has_won = env.move_enemy()
            if enemy_has_won:
                break

        score = env.score
        score_total += score
        if verbose:
            print("Episode:", i + 1, "Score:", score)
            print("Average score:", score_total / (i + 1))

    print("Average score over", episodes, "episodes:", score_total / episodes)
    env.stats()
    return score_total / episodes


def test_random(delay=0.0, display=False, episodes=100, verbose=False, test=True):
    from random import randint

    env = Env(test=test)
    score_total = 0
    print("Testing random...")

    for i in range(episodes):
        env.reset()

        if display:
            env.render()
            sleep(delay)

        while True:
            move_done, build_done = False, False
            # get old state
            env.get_move_state()

            # === Move
            # perform move
            move_number = randint(0, env.get_action_size() - 1)
            move_choice = [0] * env.get_action_size()
            move_choice[move_number] = 1
            move_reward, move_done = env.move(move_choice)
            if display:
                print()
                print("Move choice:", move_choice)
                env.render()
                print("Move reward:", move_reward)
                print("Move done:", move_done)
                print()
                if not move_done:
                    sleep(delay)

            if not move_done:
                env.get_build_state()

                # === Build
                # perform build
                build_number = randint(0, env.get_action_size() - 1)
                build_choice = [0] * env.get_action_size()
                build_choice[build_number] = 1
                build_reward, build_done = env.build(build_choice)
                if display:
                    print("Build choice:", build_choice)
                    env.render()
                    print("Build reward:", build_reward)
                    print("Build done:", build_done)
                    print()
                    if not build_done:
                        sleep(delay)

            if move_done or build_done:
                break

        score = env.score
        score_total += score
        if verbose:
            print("Episode:", i + 1, "Score:", score)
            print("Average score:", score_total / (i + 1))

    print("Average score over", episodes, "episodes:", score_total / episodes)
    env.stats()
    return score_total / episodes


def test_no_mistake(delay=0.0, display=False, episodes=100, verbose=False, test=True):
    from random import randint, choice

    env = Env(test=test)

    def get_random_action_from_state(state):
        possible_actions = state[-env.get_action_size() :]
        possible_action_numbers = []
        for i in range(len(possible_actions)):
            if possible_actions[i] == 1:
                possible_action_numbers.append(i)
        if len(possible_action_numbers) == 0:
            action = 0
        else:
            action = choice(possible_action_numbers)
        actions = [0] * env.get_action_size()
        actions[action] = 1
        return actions

    score_total = 0
    print("Testing No mistakes...")

    for i in range(episodes):
        env.reset()

        if display:
            env.render()
            sleep(delay)

        while True:
            move_done, build_done = False, False
            # get old state
            move_state = env.get_move_state()

            # === Move
            # perform move
            move_choice = get_random_action_from_state(move_state)
            move_reward, move_done = env.move(move_choice)
            if display:
                print()
                print("Move choice:", move_choice)
                env.render()
                print("Move reward:", move_reward)
                print("Move done:", move_done)
                print()
                if not move_done:
                    sleep(delay)

            if not move_done:
                build_state = env.get_build_state()

                # === Build
                # perform build
                build_choice = get_random_action_from_state(build_state)
                build_reward, build_done = env.build(build_choice)
                if display:
                    print("Build choice:", build_choice)
                    env.render()
                    print("Build reward:", build_reward)
                    print("Build done:", build_done)
                    print()
                    if not build_done:
                        sleep(delay)

            if move_done or build_done:
                break

        score = env.score
        score_total += score
        if verbose:
            print("Episode:", i + 1, "Score:", score)
            print("Average score:", score_total / (i + 1))

    print("Average score over", episodes, "episodes:", score_total / episodes)
    env.stats()
    return score_total / episodes


if __name__ == "__main__":
    # Profiling
    import pstats
    import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()

    # === Small model tests
    test(delay=0.6, display=True, episodes=1, test=True)
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
